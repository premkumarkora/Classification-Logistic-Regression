import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df = pd.read_csv("../../data/diabetes.csv")
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = df['Outcome']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)
predict = RFC.predict(x_test)

print(classification_report(y_test,predict))
print("************")
RFC = DecisionTreeClassifier()
RFC.fit(x_train,y_train)
predict = RFC.predict(x_test)
print(classification_report(y_test,predict))





