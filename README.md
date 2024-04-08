Data Science
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


Information Retrieval

#practical 1
#Write a python program to demonstrate bitwise operation. 

def bitwise_operations(a, b):
    bitwise_and = a & b  # Bitwise AND operation
    print("bitwise AND:", bitwise_and)    
    bitwise_or = a | b  # Bitwise OR operation
    print(f"bitwise OR: {bitwise_or}")
    bitwise_xor = a ^ b  # Bitwise XOR operation
    print(f"bitwise XOR: {bitwise_xor}")
    bitwise_not_a = ~a  # Bitwise NOT operation for a
    print(f"bitwise NOT of a: {bitwise_not_a}")
    bitwise_not_b = ~b  # Bitwise NOT operation for b
    print(f"bitwise NOT of b: {bitwise_not_b}")
    bitwise_left = a << 1  # Bitwise LEFT-Shift operation
    print(f"bitwise LEFT: {bitwise_left}")
    bitwise_right = a >> 2  # Bitwise RIGHT-shift operation
    print(f"bitwise RIGHT: {bitwise_right}")
a = int(input("Enter the value of a: "))
b = int(input("Enter the value of b: "))
bitwise_operations(a, b)

#practical 1
#method 2 Term Incidence Matrix
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
print("Boolean Retrieval Model Using Bitwise operations on Term Document Incidence Matrix")
corpus = {'this is the first document',
          'this document is the second document',
          'and this is the third document',
          'is this the first document?'}
print("The corpus is: \n", corpus)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus)
df = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
print("The generated dataframe:")
print(df)
print("Query processing on the term document incidence matrix")
# AND
print("Find all document ids for query 'this' AND 'first'")
alldata = df[(df['this'] == 1) & (df['first'] == 1)]
print("Document ids where 'this' AND 'first' are present are:", alldata.index.tolist())
# OR
print("Find all document ids for query 'this' OR 'first'")
alldata = df[(df['this'] == 1) | (df['first'] == 1)]
print("Document ids where 'this' OR 'first' are present are:", alldata.index.tolist())
# NOT
print("Find all document ids for query 'and' is not present")
alldata = df[(df['and'] != 1)]
print("Document ids where 'and' term is not present are:", alldata.index.tolist())
# XOR
print("Find all document ids for query 'this' XOR 'first'")
alldata = df[(df['this'] == 1) ^ (df['first'] == 1)]
print("Document ids where 'this' XOR 'first' are present are:", alldata.index.tolist())

#practical 2 Method 1
#AIM: Implement Page Rank Algorithm. 


import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G"])
G.add_edges_from([("G", "A"), ("A", "G"), ("B", "A"), ("A", "D"), ("D", "B"), ("A", "C"), ("C", "A"), ("D", "F"), ("F", "A"), ("E", "A")])
ppr1 = nx.pagerank(G)
print("Page rank values:", ppr1)
pos = nx.spiral_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_color="#f86e00")
plt.show()

#practical 2
#METHOD 2 
#Implementation of PageRank using NetworkX CODE 

import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
[G.add_node(k) for k in ["A", "B", "C"]]
G.add_weighted_edges_from([('A', 'B', 1), ('A', 'C', 1), ('C', 'A', 1), ('B', 'C', 1)])
ppr1 = nx.pagerank(G)
print("Page rank values:", ppr1)
pos = nx.spiral_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_color="#f86e00")
plt.show()

#practical 2
#method 3

def page_rank(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    num_pages = len(graph)
    initial_page_rank = 1.0 / num_pages
    page_ranks = {page: initial_page_rank for page in graph} 
    for _ in range(max_iterations):
        new_page_ranks = {}
        for page in graph:
            new_rank = (1 - damping_factor) / num_pages
            for link in graph:
                if page in graph[link]:
                    new_rank += damping_factor * (page_ranks[link] / len(graph[link]))
            new_page_ranks[page] = new_rank       
        convergence = all(abs(new_page_ranks[page] - page_ranks[page]) < tolerance for page in graph)             
        page_ranks = new_page_ranks
        if convergence:
            break
    return page_ranks
if __name__ == "__main__":
    example_graph = {
        'A': ['B', 'C'],
        'B': ['A'],
        'C': ['A', 'B'],
        'D': ['B']
    }
    result = page_rank(example_graph)
    for page, rank in sorted(result.items(), key=lambda x: x[1], reverse=True):
        print(f"Page: {page} - PageRank: {rank:4f}")

#practical 3
#AIM: Write a program to implement Levenshtein Distance. 

def leven(x, y):
    n = len(x)
    m = len(y)
    A = [[i + j for j in range(m + 1)] for i in range(n + 1)]
    for i in range(n):
        for j in range(m):
            A[i + 1][j + 1] = min(
                A[i][j + 1] + 1,
                A[i + 1][j] + 1,
                A[i][j] + int(x[i] != y[j])
            )
    return A[n][m]
print(leven("brap", "rap"))
print(leven("trial", "try"))
print(leven("horse", "force"))
print(leven("rose", "erode"))


#practical 4
#AIM: Write a program to Compute Similarity between two text documents
#Jaccard similarity method 1
import spacy
from spacy import en_core_web_sm
nlp = spacy.load('en_core_web_sm')
doc1 = nlp(u'Hello hi there!')
doc2 = nlp(u'Hello hi there!')
doc3 = nlp(u'Hey whatsup?')
print(doc1.similarity(doc2))
print(doc2.similarity(doc3))
print(doc1.similarity(doc3))


#practical 4
#Jaccard similarity method 2

def jaccard_Similarity(doc1, doc2):
    # Convert documents to sets of lowercase words
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split()) 
    # Calculate intersection and union of the two sets
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2) 
    # Calculate Jaccard similarity and return as a float
    return float(len(intersection)) / len(union)
# Define the documents
doc_1 = "Data is the new oil of the digital economy"
doc_2 = "Data is a new oil"
# Calculate and print the Jaccard similarity
print(jaccard_Similarity(doc_1, doc_2))

#Practical 4
#METHOD 3: COSINE SIMILARITY 


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
doc_1 = "Data is the new oil of the digital economy"
doc_2 = "Data is a new oil"
data = [doc_1, doc_2]
# Initialize TfidfVectorizer
Tfidf_vect = TfidfVectorizer()
# Transform the data into TF-IDF matrix
vector_matrix = Tfidf_vect.fit_transform(data)
# Get the tokens (features)
tokens = Tfidf_vect.get_feature_names_out()
# Calculate cosine similarity matrix
cosine_similarity_matrix = cosine_similarity(vector_matrix)
# Print the cosine similarity matrix along with the document names
print(cosine_similarity_matrix, ['doc_1', 'doc_2'])


#practical 5
#AIM: Write a Map Reduce Program to count the number of occurrences of each alphabetic character in a given dataset

from functools import reduce
from collections import defaultdict
def mapper(data):
    char_count = defaultdict(int)
    for char in data:
        if char.isalpha():
            char_count[char.lower()] += 1
    return char_count.items()
def reducer(counts1, counts2):
    merged_counts = defaultdict(int)
    for char, count in counts1:
        merged_counts[char] += count
    for char, count in counts2:
        merged_counts[char] += count
    return merged_counts.items()
if __name__ == "__main__":
    dataset = "Hello, world! This is a MapReduce example."
    chunks = [chunk for chunk in dataset.split()]
    # Map phase
    mapped_result = map(mapper, chunks)
    # Reduce phase
    final_counts = reduce(reducer, mapped_result)
    # Output
    for char, count in final_counts:
        print(f"Character: {char}, Count: {count}")


#practical 6
#AIM: HITS Algorithm 

import networkx as nx
# Step 2: Create a graph and add edges
G = nx.DiGraph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
# Step 3: Calculate the HITS scores
authority_scores, hub_scores = nx.hits(G)
# Step 4: Print the scores
print("Authority Scores:", authority_scores)
print("Hub Scores:", hub_scores)


#practical 7
#AIM: Write a python program for pre-processing of text document stopword removal. 
#step 1

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(set(stopwords.words('english')))
import nltk


#practical 7
#step 2


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example = "This is a sample sentence, showing off the stopwords filtration."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
print("Original Sentence:", word_tokens)
print("Filtered Sentence:", filtered_sentence)


#practical 8
#AIM:Write a program for mining twitter to identify tweets for a specific period and identify trends and named entities
#step 1

import pandas as pd
from ntscraper import Nitter
scraper = Nitter()
tweets = scraper.get_tweets('narendramodi', mode='user', number=5)

#practical 8
#step 2

final_tweets = []
for tweet in tweets['tweets']:
    data = [tweet['link'], tweet['text'], tweet['date'], tweet['stats']['likes'], tweet['stats']['comments']]
    final_tweets.append(data)
df = pd.DataFrame(final_tweets, columns=['Link', 'Text', 'Date', 'Likes', 'Comments'])
print(df)

#practical 8
#step 3

data = pd.DataFrame(final_tweets, columns=['link', 'text', 'date', 'Number of likes', 'Number of tweets'])
print(data)

#practical 9
#AIM: Write a program to implement simple web crawling. 

import requests
from parsel import Selector
import time
start = time.time()
response = requests.get('http://recurship.com/')
selector = Selector(response.text)
href_links = selector.xpath('//a/@href').getall()
image_links = selector.xpath('//img/@src').getall()
print("***********Href_links***********")
print(href_links)
print("***********/href_links***********")
print(image_links)
print("***********/image_links***********")
end = time.time()
print("Time Taken in seconds:", (end - start))


#practical 10
#AIM: Write a python program to parse XML text, generate Web graph and compute topic specific page rank.  

import xml.etree.ElementTree as ET
import networkx as nx
def parse_xml(xml_text):
    root = ET.fromstring(xml_text)
    return root
def generate_web_graph(xml_root):
    G = nx.DiGraph()
    for page in xml_root.findall('.//page'):
        page_id = page.find('id').text
        G.add_node(page_id)
        links = page.findall('.//link')
        for link in links:
            target_page_id = link.text
            G.add_edge(page_id, target_page_id)
    return G
def compute_topic_specific_pagerank(graph, topic_nodes, alpha=0.85, max_iter=100, tol=1e-6):
    personalization = {node: 1.0 if node in topic_nodes else 0.0 for node in graph.nodes}
    return nx.pagerank(graph, alpha=alpha, personalization=personalization, max_iter=max_iter, tol=tol)
if __name__ == "__main__":
    xml_data = """
    <webgraph>
        <page>
            <id>1</id>
            <link>2</link>
            <link>3</link>
        </page>
        <page>
            <id>2</id>
            <link>1</link>
            <link>3</link>
        </page>
        <page>
            <id>3</id>
            <link>1</link>
            <link>2</link>
        </page>
    </webgraph>"""
    xml_root = parse_xml(xml_data)
    web_graph = generate_web_graph(xml_root)
    topic_specific_pagerank = compute_topic_specific_pagerank(web_graph, topic_nodes=['1', '2'])
    print("Topic Specific PageRank:")
    for node, score in sorted(topic_specific_pagerank.items(), key=lambda x: x[1], reverse=True):
        print(f"Node: {node} - PageRank: {score:.4f}")





