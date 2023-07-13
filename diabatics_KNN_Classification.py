import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df=pd.read_csv("../../data/diabetes.csv")

x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']
x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.7)

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

y_train_pred = knn.predict(x_train)
y_test_pred = knn.predict(x_test)

print("Testing accuracy: ",accuracy_score(y_test,y_test_pred))
print("Training Accuracy: ",accuracy_score(y_train,y_train_pred))
print(classification_report(y_test,y_test_pred))
