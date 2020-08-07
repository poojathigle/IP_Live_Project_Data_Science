"""
Author : Pooja Thigle		Visualization Model

PROBLEM STATEMENT
TECHNOLOGY: DATA SCIENCE

Internal link to data set :
https://cloudcounselage24.bitrix24.com/disk/showFile/39278/?&ncc=1&ts=1592284132&filename=DS_DATESET.csv

Students from different cities from the state of Maharashtra had applied for the Cloud
Counselage Internship Program. We have the dataset of consisting information of all the
students. Using this data we want to get more insights and draw out more meaningful
conclusions. Interns are expected to build a data visualization model and find the best data
segmentation model using the student’s dataset. Following are the tasks interns need to
perform
"""

import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('Visualization-output.pdf')

# load data set
# dataset = pd.read_csv('E:\CC_Internship\Live Project\\DS_DATESET.csv')
# dataset = pd.read_csv(input("Enter path to dataset: ").strip())
data = sys.argv[1]
dataset = pd.read_csv(data, header=0, escapechar='\\')
df = dataset.copy()
sns.set(style="darkgrid")
plt.figure(figsize=(10, 5))


"""
a. The number of students applied to different technologies.
"""

"""Ans:-----------------------------------------------------------------------------------------"""
sns.countplot(x='Areas of interest', data=df, palette="Set1")
plt.tight_layout()
pdf.savefig()
plt.close()

# plt.show()


"""
b. The number of students applied for Data Science who knew "Python” and
who didn’t.
"""
"""Ans:-----------------------------------------------------------------------------------------"""
options = df.loc[(df["Areas of interest"] == "Data Science "),
                 ['Areas of interest', 'Programming Language Known other than Java (one major)']]
# print(len(options))

# print(options['Programming Language Known other than Java (one major)'].unique().tolist())

options['Programming Language Known other than Java (one major)'].replace(
    ['HTML/CSS', 'PHP', 'C', 'C++', '.Net', 'JavaScript', 'C#'],
    ['Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other'], inplace=True)

fig1 = sns.countplot(x="Areas of interest", hue="Programming Language Known other than Java (one major)", data=options)
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()


"""
c. The different ways students learned about this program.
"""

"""Ans:-----------------------------------------------------------------------------------------"""

plt.figure(figsize=(8, 5))
fig2 = df['How Did You Hear About This Internship?'].value_counts().plot(kind='pie')       # Pie plot
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()

"""
d. Students who are in the fourth year and have a CGPA greater than 8.0.
"""
"""Ans:-----------------------------------------------------------------------------------------"""
f = df.loc[(df["CGPA/ percentage"] >= 8.00),
           ['Which-year are you studying in?', 'CGPA/ percentage']]

fig3 = sns.countplot(y="Which-year are you studying in?", palette='hls', data=f)      # palette = hls, coolwarm, rainbow,set1
plt.title("Students with CGPA > 8")
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()


"""
e. Students who applied for Digital Marketing with verbal and written
communication score greater than 8.
"""
"""Ans:-----------------------------------------------------------------------------------------"""

datam = df[df['Rate your verbal communication skills [1-10]'] > 8]
df1 = datam[['Areas of interest', 'Rate your verbal communication skills [1-10]',
             'Rate your written communication skills [1-10]']]
df2 = df1[df1['Rate your written communication skills [1-10]'] > 8]

dm = df[df['Areas of interest'] == 'Digital Marketing ']
dm = dm[['Areas of interest', 'Rate your verbal communication skills [1-10]',
         'Rate your written communication skills [1-10]']]                       # final data frame dm
dmplot = sns.scatterplot(x="Rate your verbal communication skills [1-10]",
                         y="Rate your written communication skills [1-10]", hue="Areas of interest", data=dm)
plt.title("Digital Marketing interns with verbal and written communication score > 8")
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()

"""
f. Year-wise and area of study wise classification of students.
"""
"""Ans:-----------------------------------------------------------------------------------------"""

graph6 = sns.scatterplot(y="Which-year are you studying in?",
                         x="Major/Area of Study", palette='coolwarm', data=df)
plt.title("Year-wise and area of study wise classification of students")
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()

"""
g. City and college wise classification of students.
"""
"""Ans:-----------------------------------------------------------------------------------------"""
plt.figure(figsize=(10, 5))
graph7 = sns.scatterplot(x="City",
                         y="College name", palette='coolwarm', data=df)
plt.title("City and college wise distribution")
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()


"""
h. Plot the relationship between the CGPA and the target variable.
"""
"""Ans:-----------------------------------------------------------------------------------------"""
graph8 = sns.scatterplot(x="CGPA/ percentage", y="Degree",
                         hue="Label", palette='hls', data=df)
plt.title("relationship between the CGPA and the target variable")
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()


"""
i. Plot the relationship between the Area of Interest and the target variable.
"""
"""Ans:-----------------------------------------------------------------------------------------"""

graph9 = sns.scatterplot(y="Areas of interest", x="Label", hue="Degree", palette='hls', data=df)
plt.title("Relationship between the Area of Interest and the target variable")
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()


"""
j. Plot the relationship between the year of study, major, and the target variable."""
"""Ans:-----------------------------------------------------------------------------------------"""

graph10 = sns.scatterplot(y="Which-year are you studying in?", x="Major/Area of Study", hue="Label", palette='hls',
                          data=df)
plt.title("Relationship between the year of study, major, and the target variable")
plt.tight_layout()
pdf.savefig()
plt.close()
# plt.show()


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
# print("Confusion Matrix:")
# print(result)
result1 = classification_report(y_test, y_pred)
# print("Classification Report:",)
# print(result1)
result2 = accuracy_score(y_test, y_pred)
# print("Accuracy: ", result2)
# print("F1 Score: ", metrics.f1_score(y_test, y_pred))
print()

# Visualize result using HeatMap

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
