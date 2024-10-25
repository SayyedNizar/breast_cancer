import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

data = pd.read_csv('D:/breast-cancer.csv')  #Loading the dataset

#Data Preprocessing 
data.duplicated().sum()  
data.describe().T
data.isnull().sum()    #Handling Missing Values


data = pd.get_dummies(data, drop_first=True)  #Categorical Data

#Splitting Features and Targets
x = data.drop(columns=['diagnosis_M'])  
y = data['diagnosis_M']

#Normalizaton
scaler = StandardScaler()    #Calculates mean and Standard Deviation
x_scaled = scaler.fit_transform(x)
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)


print(x_scaled_df.head())
print(data.columns)
print(data.head())

#Outlier Detection
def detect_outliers_iqr(df):
    outliers = pd.DataFrame()
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):  # Only apply to numeric columns
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Identify outliers
            outliers_in_column = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outliers = pd.concat([outliers, outliers_in_column], axis=0)
    return outliers.drop_duplicates()

outliers = detect_outliers_iqr(data)
print(f"Number of outliers detected: {len(outliers)}")

#Removing Outliers 
outliers = outliers[outliers.index.isin(data.index)] 
data=data.drop(outliers.index,axis=0)
print(data)

#Training and Testing the Data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
x_train_df = pd.DataFrame(x_train, columns=x.columns)
x_test_df = pd.DataFrame(x_test, columns=x.columns)


print(x_train_df.head())
print(y_train.head())
print(x_test_df.head())
print(y_test.head())
print(y.value_counts())
data.info()

#Balancing the Data
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#Visualization Of Data
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Class distribution before SMOTE")
plt.xlabel("Class (0: Benign, 1: Malignant)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution after SMOTE")
plt.xlabel("Class (0: Benign, 1: Malignant)")
plt.ylabel("Count")
plt.show()


base_classifier = DecisionTreeClassifier(max_depth=1)
ada_boost = AdaBoostClassifier(estimator=base_classifier, n_estimators=50, algorithm='SAMME', random_state=42)
ada_boost.fit(x_train_resampled, y_train_resampled)


y_pred = ada_boost.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



