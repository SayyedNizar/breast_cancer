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

data = pd.read_csv('D:/breast-cancer.csv')

data.duplicated().sum()
data.describe().T
data.isnull().sum()


data = pd.get_dummies(data, drop_first=True)


x = data.drop(columns=['diagnosis_M'])
y = data['diagnosis_M']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)


print(x_scaled_df.head())
print(data.columns)
print(data.head())


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
x_train_df = pd.DataFrame(x_train, columns=x.columns)
x_test_df = pd.DataFrame(x_test, columns=x.columns)


print(x_train_df.head())
print(y_train.head())
print(x_test_df.head())
print(y_test.head())
print(y.value_counts())
data.info()


smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)


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


input_data = np.array([[12.34, 14.82, 78.32, 477.1, 0.1018, 0.1010, 0.0923, 0.066, 0.1952, 0.0564, 
                        0.75, 1.25, 5.0, 400.0, 0.012, 0.020, 0.030, 0.004, 0.015, 0.002, 
                        13.00, 20.00, 100.0, 600.0, 0.150, 0.210, 0.240, 0.045, 0.170, 0.030]])

scaled_input = scaler.transform(input_data)


prediction = ada_boost.predict(scaled_input)
print(f"Predicted Class (0: Benign, 1: Malignant): {prediction[0]}")


probability = ada_boost.predict_proba(scaled_input)
print(f"Prediction Probability: {probability}")


actual_label = y_test.iloc[0]  
print(f"Actual Label: {actual_label}")
print(f"Predicted Class (0: Benign, 1: Malignant): {int(prediction[0])}")
print(f"Prediction Probability: {probability[0]}")

if int(prediction[0]) == actual_label:
    print("The prediction is correct.")
else:
    print("The prediction is incorrect.")
