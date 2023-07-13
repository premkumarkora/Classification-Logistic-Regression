import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("../../data/SUV purchase prediction.csv") 

#sns.countplot(x="Purchased", hue="Gender", data=df)

df= df.drop(['User ID'], axis=1)
sumOfIsna= df.isna().sum()
Gender = pd.get_dummies(df['Gender'],drop_first=True)
df = pd.concat([df,Gender],axis=1)

X = df[['Age','EstimatedSalary','Male']]
Y = df['Purchased']
x_train,x_test,y_train,y_test = 
            train_test_split(X,Y,test_size=0.25, random_state=0)


standardscaler = StandardScaler()
x_train = standardscaler.fit_transform(x_train)
x_test = standardscaler.transform(x_test)

log = LogisticRegression(random_state=0)
log.fit(x_train,y_train)
y_predicted = log.predict(x_test)
print("accuracy_score :", accuracy_score(y_test,y_predicted))
print("Classification Report:")
print( classification_report(y_test,y_predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_predicted))