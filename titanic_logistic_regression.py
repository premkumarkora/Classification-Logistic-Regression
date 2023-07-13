import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


titanic_data = pd.read_csv("../../data/titanic.csv")
titanic_data= titanic_data.drop(['Cabin'], axis=1)
titanic_data = titanic_data.dropna()
#x= titanic_data.isna().sum() 

#sns.countplot(x="Survived", hue="Sex", data=titanic_data)
#sns.countplot(x="Survived", hue="Pclass", data=titanic_data)
#sns.countplot(x="Survived", data=titanic_data)
#titanic_data["Age"].plot.hist()
#titanic_data["Fare"].plot.hist(bins=20)

gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
Embarked = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
Pclass = pd.get_dummies(titanic_data['Pclass'],drop_first=True)
titanic_data = pd.concat([titanic_data,gender,Embarked,Pclass],axis=1)

#x=titanic_data.drop('Survived', axis=1)
X = titanic_data[['Age','SibSp','Parch','Fare','Q','male','S',2, 3]]
Y = titanic_data['Survived']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3, random_state=1)
log = LogisticRegression()
log.fit(x_train,y_train)
predicted = log.predict(x_test)
print("accuracy_score :")
print(accuracy_score(y_test,predicted))
print("Classification Report:")
print( classification_report(y_test,predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test,predicted))
# Curve will be sigmoid curve


#sns.pairplot(titanic_data[['Survived','Age','SibSp','Parch','Fare','Q','male']])
#plt.show()








