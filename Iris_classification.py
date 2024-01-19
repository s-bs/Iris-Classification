import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load dataset
data =pd.read_csv("iris.csv")

# separate variables 
X=data.drop('variety',axis=1)
Y=data['variety']

# splitt dataset in train and test
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# START Logistic
model_logistic =LogisticRegression()
model_logistic.fit(X_train,Y_train)
logestic_predict=model_logistic.predict(X_test)
accuracy2=accuracy_score(Y_test,logestic_predict)
print('Logistic Accuracy',accuracy2)
# End Logistic

# Start SVM
model_svm = SVC()
model_svm.fit(X_train,Y_train)
svm_predict =model_svm.predict(X_test)
accuracy3=accuracy_score(Y_test,svm_predict)
print('SVM Accuracy',accuracy3)
# END SVM

# Start Decision tree
model_Dtree =DecisionTreeClassifier()
model_Dtree.fit(X_train,Y_train)
Dtree_predict =model_Dtree.predict(X_test)
accuracy4=accuracy_score(Y_test,Dtree_predict)
print('DECISION-TREE Accuracy',accuracy4)
# End Decision tree

# Start Random Forest
model_RFA=RandomForestClassifier()
model_RFA.fit(X_train,Y_train)
RFA_predict=model_RFA.predict(X_test)
accuracy5=accuracy_score(Y_test,RFA_predict)
print('Random Forest Accuracy',accuracy5)
# End Random forest