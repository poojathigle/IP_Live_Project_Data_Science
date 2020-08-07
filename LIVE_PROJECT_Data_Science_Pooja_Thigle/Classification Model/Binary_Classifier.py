"""
@author : Pooja Thigle

Classification model : Random Forest
"""
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('Visualization-output.pdf')

# load data set
# dataset = pd.read_csv('C:\\Users\\hp\\DS_DATESET.csv')
# dataset = pd.read_csv(input("Enter path to dataset: ").strip())
data = sys.argv[1]
dataset = pd.read_csv(data, header=0, escapechar='\\')
#dataset = np.loadtxt(data, delimiter=' ')
df = dataset.copy()
"""
3. Identify the best binary classifier to classify data into “eligible/1” and “not eligible/0”.
"""

"""Ans:-----------------------------------------------------------------------------------------"""

# split the data-set in features and target variable

dset = df.copy()

# drop unnecessary and Null value columns
dset.drop("Certifications/Achievement/ Research papers", axis=1, inplace=True)
dset.drop("First Name", axis=1, inplace=True)
dset.drop("Last Name", axis=1, inplace=True)
dset.drop("University Name", axis=1, inplace=True)
dset.drop("DOB [DD/MM/YYYY]", axis=1, inplace=True)
dset.drop("Email Address", axis=1, inplace=True)
dset.drop("Emergency Contact Number", axis=1, inplace=True)
dset.drop("Contact Number", axis=1, inplace=True)
dset.drop("link to Linkedin profile", axis=1, inplace=True)
dset.drop("College name", axis=1, inplace=True)
dset.drop("Link to updated Resume (Google/ One Drive link preferred)", axis=1, inplace=True)
dset.drop("State", axis=1, inplace=True)


# print(dset.isnull().sum())

dset['Label'].replace(
    ['eligible', 'ineligible'],
    ['1', '0'], inplace=True)

lbl = pd.get_dummies(dset['Label'], drop_first=True)

java1 = pd.get_dummies(dset['Have you worked core Java'], drop_first=True)
sql1 = pd.get_dummies(dset['Have you worked on MySQL or Oracle database'], drop_first=True)
oops1 = pd.get_dummies(dset['Have you studied OOP Concepts'], drop_first=True)
city = pd.get_dummies(dset['City'], drop_first=True)
gender = pd.get_dummies(dset['Gender'], drop_first=True)
major = pd.get_dummies(dset['Major/Area of Study'], drop_first=True)
interest = pd.get_dummies(dset['Areas of interest'], drop_first=True)
source = pd.get_dummies(dset['How Did You Hear About This Internship?'], drop_first=True)
languages = pd.get_dummies(dset['Programming Language Known other than Java (one major)'], drop_first=True)


# concat new dummy columns to dset
dset = pd.concat([dset, city, gender, major, source, languages, interest, java1, lbl], axis=1)
dset.rename(columns={'1': 'label'}, inplace=True)
dset.rename(columns={'Yes': 'Java'}, inplace=True)

dset = pd.concat([dset, sql1], axis=1)
dset.rename(columns={'Yes': 'sql'}, inplace=True)

dset = pd.concat([dset, oops1], axis=1)
dset.rename(columns={'Yes': 'oops'}, inplace=True)

dset.drop("Label", axis=1, inplace=True)
dset.drop("Have you worked core Java", axis=1, inplace=True)
dset.drop("Have you worked on MySQL or Oracle database", axis=1, inplace=True)
dset.drop("Have you studied OOP Concepts", axis=1, inplace=True)
dset.drop("City", axis=1, inplace=True)
dset.drop("Gender", axis=1, inplace=True)
dset.drop("How Did You Hear About This Internship?", axis=1, inplace=True)
dset.drop("Course Type", axis=1, inplace=True)
dset.drop("Programming Language Known other than Java (one major)", axis=1, inplace=True)

# for col in dset.columns:
#     print(col)

feature_cols = ['CGPA/ percentage', 'Expected Graduation-year', 'Java', 'sql', 'oops', 'Mumbai',
                'NaviMumbai', 'Pune', 'Sangli', 'Solapur', 'Male', 'Electrical Engineering',
                'Electronics and Telecommunication', 'Rate your written communication skills [1-10]',
                'Rate your verbal communication skills [1-10]', 'Ex/Current Employee', 'Facebook', 'Friend', 'Intern',
                'LinkedIn', 'Newspaper', 'Other', 'Twitter', 'C', 'C#', 'C++', 'HTML/CSS', 'JavaScript', 'PHP',
                'Python', 'Big Data ', 'Blockchain ', 'Cloud Computing ', 'Cyber Security ', 'Data Science ',
                'DevOps ', 'Digital Marketing ', 'Information Security', 'IoT ', 'Machine Learning', 'Mobility',
                'Python ', 'QMS/Testing ', 'RPA ', 'Web Development ']

X = dset[feature_cols]  # Features
# print(X.head(10))
y = dset.label  # Target variable
# print(feature_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
result = confusion_matrix(y_test, y_pred)
# print()
# print("Confusion Matrix:")
# print(result)
result1 = classification_report(y_test, y_pred)
# print("Classification Report:",)
# print(result1)
result2 = accuracy_score(y_test, y_pred)
# print("Accuracy: ", result2)
print(metrics.f1_score(y_test, y_pred))


# Visualize result using HeatMap
"""
class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create Heatmap
sns.heatmap(pd.DataFrame(result), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
pdf.savefig()
# plt.show()

d = pdf.infodict()
d['Title'] = 'Visualization-output.pdf'
d['Author'] = 'Pooja Thigle'
print("Visualization-output.pdf of Graphs generated.")
pdf.close()
plt.close()
"""
# detail Output:
"""

C:\hp\AppData\Local\Programs\Python\Python38-32\python.exe E:/PythonProjects/pooja_thigle.py
Confusion Matrix:
[[1811    0]
 [   0 1189]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1811
           1       1.00      1.00      1.00      1189

    accuracy                           1.00      3000
   macro avg       1.00      1.00      1.00      3000
weighted avg       1.00      1.00      1.00      3000

Accuracy:  1.0
F1 Score:  1.0

Process finished with exit code 0
"""