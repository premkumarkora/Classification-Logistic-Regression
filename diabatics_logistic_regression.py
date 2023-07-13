import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df = pd.read_csv("../../data/diabetes.csv")
x= df.drop(["Outcome"], axis=1)
y= df["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
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










