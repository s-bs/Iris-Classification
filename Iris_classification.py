import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load dataset
data =pd.read_csv("iris.csv")
# visualiz all data
sns.pairplot(data,hue='variety')
# separate variables 
X=data.drop('variety',axis=1)
Y=data['variety']
# splitt dataset in train and test
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

def train_test(classifier,X_train,Y_train,X_test,Y_test):
    classifier.fit(X_train,Y_train)
    predict = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test,predict)
    return predict,accuracy

# Classification Algorithms
classifiers =[LogisticRegression(),SVC(),DecisionTreeClassifier(),RandomForestClassifier()]
    
for classifier in classifiers:
    predict,accuracy=train_test(classifier,X_train,Y_train,X_test,Y_test)
    print(classifier,accuracy)

plt.show()