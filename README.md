#practical 1.Write a python program to plot word cloud for a wikipedia page of any topic. 

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import wikipedia as wp
result = wp.page("Computer Science")
final_result = result.content
def plot_wordcloud(wc):
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
wc = WordCloud(width=500, height=500, background_color="cyan", random_state=10, stopwords=STOPWORDS).generate(final_result)
wc.to_file("cs.png")
plot_wordcloud(wc)

PRACTICAL 2 
AIM: Write a python program to perform Web Scrapping 
1.HTML scrapping- use Beautiful Soup

import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
url = "https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area"
page = urlopen(url)
html_page = page.read().decode("utf-8")
soup = BeautifulSoup(html_page, "html.parser")
table = soup.find("table")
SrNo = []
Country = []
Area = []
rows = table.find("tbody").find_all("tr")
for row in rows:
    cells = row.find_all("td")
    if cells:
        SrNo.append(cells[0].get_text().strip("\n"))  # strip extra characters
        Country.append(cells[1].get_text().strip("\xa0").strip("\n").strip("\[2]*"))  # strip extra characters
        Area.append(cells[2].get_text().strip("\n*").replace("\n", "").replace("•", ""))  # strip extra characters
countries_df = pd.DataFrame()
countries_df["SrNo"] = SrNo
countries_df["Country"] = Country
countries_df["Area"] = Area
print(countries_df.head(10))

2.JSON scrapping 

import pandas as pd
import urllib,json
url = "https://jsonplaceholder.typicode.com/users"
response = urllib.request.urlopen(url)
data = json.loads(response.read())
print(data)
id1 = []
username = []
email = []
for item in data:
    if "id" in item.keys():  # check if 'id' is present in dictionary
        id1.append(item["id"])
    else:
        id1.append("NA")

    if "username" in item.keys():  # check if 'username' is present in dictionary
        username.append(item["username"])
    else:
        username.append("NA")

    if "email" in item.keys():  # check if 'email' is present in dictionary
        email.append(item["email"])
    else:
        email.append("NA")
print(id1)
print(username)
print(email)


import pandas as pd
user_accounts = pd.DataFrame()
user_accounts["USER ID"] = id1
user_accounts["USERNAME"] = username
user_accounts["EMAIL"] = email
print(user_accounts.head(10))

PRACTICAL 4 
AIM: Exploratory data analysis in Python using Titanic Database

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
titanic = pd.read_csv("C:/sem 6 notes/ds practical/train.csv")
print(titanic.head())
print(titanic.info())  
print(titanic.describe())  
print(titanic.isnull().sum())
titanic_cleaned = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(titanic_cleaned.info())
sns.catplot(x='Sex', hue='Survived', kind='count', data=titanic_cleaned)
grp1 = titanic_cleaned.groupby(['Sex', 'Survived'])['Survived'].count()
gender_survived = grp1.unstack()
sns.heatmap(gender_survived, annot=True, fmt="d")
plt.show()

grpl = titanic_cleaned.groupby(["Pclass", "Survived"])
gender_survived = grpl.size().unstack()
sns.heatmap(gender_survived, annot=True, fmt="d")

import pandas as pd
import seaborn as sns
sns.violinplot(x="Sex", y="Age", hue="Survived", data=titanic_cleaned, split=True)
print("Oldest person on board: ", titanic_cleaned['Age'].max())
print("Oldest person on board: ", titanic_cleaned['Age'].min())
print("Oldest person on board: ", titanic_cleaned['Age'].mean())
def impute(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 34
        elif Pclass == 2:
            return 29 
        else:
            return 24
    else:
        return Age
titanic_cleaned['Age'] = titanic_cleaned[['Age', 'Pclass']].apply(impute, axis=1)
print(titanic_cleaned.isnull().sum())
titanic_cleaned.corr(method='pearson')

PRACTICAL 5 
AIM: Exploratory data analysis in Python using Titanic Dataset 
1)Write a python program to build a regression model that could predict the 
salary of an employee from the given experience and visualize univariate linear 
regression on it.

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x, y, coef = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)
x = np.interp(x, (x.min(), x.max()), (0, 20))
print(len(x))
y = np.interp(y, (y.min(), y.max()), (20000, 150000))
print(len(y))
plt.plot(x, y,'.', label="Training data")
plt.xlabel("Experience (in years)")
plt.ylabel("Salary (in rupees)")
plt.title("Experience vs Salary")
plt.show()

from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(x, y)
y_pred = reg_model.predict(x)
plt.plot(x, y_pred, color="black")
plt.plot(x, y,'.', label="Training data")
plt.xlabel("Experience (in years)")
plt.ylabel("Salary (in rupees)")
plt.title("Experience vs Salary")

data = {'Experience': np.round(x.flatten()), 'Salary': np.round(y)}
df = pd.DataFrame(data)
print(df.head(10))

x1=[[13.0]]
y1=reg_model.predict(x1)
print(np.round(y1,2))

2) Write a python program to simulate linear model Y=10+7*x+e for random 100 
samples and visualize univariate linear regression on it.

from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
import numpy as np 
reg_model1 = LinearRegression() 
x = np.random.rand(100,1) 
yintercept = 10 
slope = 7 
error = np.random.rand(100, 1) 
y = yintercept + (slope * x) + error
reg_model1.fit(x,y) 
y_pred = reg_model1.predict(x)
plt.scatter(x, y, s=10) 
plt.xlabel("x") 
plt.ylabel("Y") 
plt.title("Equation Regression Model") 
plt.plot(x, y_pred, color="black") 

PRACTICAL 6 
AIM: Write a python program to show multivariate regression model. 

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
boston = pd.read_csv("C:/sem 6 notes/ds practical/Boston.csv")
boston.head()

import pandas as pd
boston_x = pd.DataFrame(boston.iloc[:, :13])  
boston_y = pd.DataFrame(boston.iloc[:, -1])   
boston_y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boston_x, boston_y, test_size=0.3)
print("xtrain Shape:", x_train.shape)
print("ytrain Shape:", y_train.shape)
print("xtest Shape:", x_test.shape)
print("ytest Shape:", y_test.shape)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)
y_pred_linear = regression.predict(x_test)
y_pred_df = pd.DataFrame(y_pred_linear, columns=["Predicted"])
y_pred_df.head()

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_linear, c="blue")
plt.xlabel("Actual Price (medv)")
plt.ylabel("Predicted Price (medv)")
plt.title("Actual vs Predicted")
plt.show()

PRACTICAL 7 
AIM: Write a python program to implement KNN algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
breast_cancer_df = load_breast_cancer()
x = pd.DataFrame(breast_cancer_df.data, columns=breast_cancer_df.feature_names)
x.head()
x = x[["mean area", "mean compactness"]]
x.head()

y = pd.Categorical.from_codes(breast_cancer_df.target, breast_cancer_df.target_names)
print(y)

y = pd.get_dummies(y, drop_first=True)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x, y, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(x_train, Y_train)

import seaborn as sns
sns.set()
sns.scatterplot(x="mean area", y="mean compactness", hue="benign", data=x_test.join(Y_test, how="outer"))

import matplotlib.pyplot as plt
y_pred = knn.predict(x_test)
plt.scatter(x_test["mean area"], x_test["mean compactness"], c=y_pred, cmap="coolwarm", alpha=0.7)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
labels = np.asarray(labels).reshape(2, 2)
categories = ["Zero", "One"]
ax = plt.subplot()
cf = [[42, 13], [9, 79]]
sns.heatmap( cf,annot=True, ax=ax)
ax.set_xlabel("Predicted values")
ax.set_ylabel("Actual values")
ax.set_title("Confusion Matrix")

from sklearn.metrics import confusion_matrix
tp, fn, fp, tn = confusion_matrix(Y_test, y_pred, labels=[1, 0]).reshape(-1)
print("Values of TP, FN, FP, TN:", tp, fn, fp, tn)

accuracy = (tp + tn) / (tp + tn + fp + fn)
print("Accuracy:", accuracy)
precision = tp / (tp + fp)
print("Precision:", precision)
recall = tp / (tp + fn)
print("Recall:", recall)

from sklearn.metrics import f1_score
f1_score(Y_test, y_pred)

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, y_pred)


practical 8-mongodb

1)1.Create a database Institution ,Create a Collection Staff and Insert ten 
documents in it with fields: empid,empname,salary and designation.

use Institution
db.Staff.insertMany([   {"empid": 1, "empname": "John", "salary": 60000, "designation": "Manager"},   {"empid": 2, "empname": "Alice", "salary": 55000, "designation": "Accountant"},   {"empid": 3, "empname": "Bob", "salary": 50000, "designation": "Manager"},   {"empid": 4, "empname": "Carol", "salary": 48000, "designation": "Accountant"},   {"empid": 5, "empname": "David", "salary": 45000, "designation": "Manager"},   {"empid": 6, "empname": "Eva", "salary": 43000, "designation": "Accountant"},   {"empid": 7, "empname": "Frank", "salary": 42000, "designation": "Manager"},   {"empid": 8, "empname": "Grace", "salary": 40000, "designation": "Accountant"},   {"empid": 9, "empname": "Harry", "salary": 38000, "designation": "Manager"},   {"empid": 10, "empname": "Ivy", "salary": 35000, "designation": "Accountant"}])
db.Staff.find({}, {"_id": 0, "empid": 1, "designation": 1})
db.Staff.find().sort({"salary": -1})
db.Staff.find({ "$or": [{"designation": "Manager"}, {"salary": {"$gt": 50000}}] })
db.Staff.updateMany({ "designation": "Accountant" }, { "$set": { "salary": 45000 } })
db.Staff.deleteMany({ "salary": { "$gt": 100000 } })



2)Create a database Institution .Create a Collection Student and Insert ten 
documents in it with fields: RollNo,Name,Class and TotalMarks(out of 500).  

use Institution
db.createCollection("Student")
db.Student.insertMany([
  { "RollNo": 1, "Name": "John Doe", "Class": "MSc", "TotalMarks": 450 },
  { "RollNo": 2, "Name": "Jane Smith", "Class": "BSc", "TotalMarks": 380 },
  { "RollNo": 3, "Name": "Alice Johnson", "Class": "MSc", "TotalMarks": 420 },
  { "RollNo": 4, "Name": "Bob Brown", "Class": "BSc", "TotalMarks": 390 },
  { "RollNo": 5, "Name": "Emily Davis", "Class": "MSc", "TotalMarks": 480 },
  { "RollNo": 6, "Name": "Michael Wilson", "Class": "BSc", "TotalMarks": 360 },
  { "RollNo": 7, "Name": "Sarah Martinez", "Class": "MSc", "TotalMarks": 410 },
  { "RollNo": 8, "Name": "David Anderson", "Class": "BSc", "TotalMarks": 500 },
  { "RollNo": 9, "Name": "Laura Taylor", "Class": "MSc", "TotalMarks": 490 },
  { "RollNo": 10, "Name": "Kevin Garcia", "Class": "BSc", "TotalMarks": 370 }
])
db.Student.find()
db.Student.find().sort({ "TotalMarks": -1 })
db.Student.find({ $or: [{ "Class": "MSc" }, { "TotalMarks": { $gt: 400 } }] })
db.Student.deleteMany({ "TotalMarks": { $lt: 200 } })







