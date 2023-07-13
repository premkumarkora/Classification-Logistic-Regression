import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
df=pd.read_csv("../../data/diabetes.csv")
#print(df)

x=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y=df['Outcome']
x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.7)

sv=SVC()
sv.fit(x_train, y_train)
y_test_predict = sv.predict(x_test)

print(classification_report(y_test,y_test_predict))