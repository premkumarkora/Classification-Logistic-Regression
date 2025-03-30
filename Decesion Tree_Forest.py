import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import  seaborn  as sns
df = sns.load_dataset("iris")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
df = pd.read_csv("../../data/diabetes.csv")
#df = df.head(200)
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
obj = DecisionTreeClassifier()
dtree = obj.fit(x_train,y_train)
predict = obj.predict(x_test)
predict_train = obj.predict(x_train)
print("Testing accuracy: ",accuracy_score(y_test,predict))
print("Training Accuracy: ",accuracy_score(y_train,predict_train))

print("************Random Forest Classifier***************")
dtc = RandomForestClassifier(n_estimators=15)
dtc.fit(x_train,y_train)
predict = dtc.predict(x_test)
predict_train = dtc.predict(x_train)
print("Testing accuracy: ",accuracy_score(y_test,predict))
print("Training Accuracy: ",accuracy_score(y_train,predict_train))


print("***********Bagging****************")
dtc = BaggingClassifier(DecisionTreeClassifier(),max_samples=1,max_features=0.6,n_estimators=25)
dtc.fit(x_train,y_train)
predict = dtc.predict(x_test)
predict_train = dtc.predict(x_train)
print("Testing accuracy: ",accuracy_score(y_test,predict))
print("Training Accuracy: ",accuracy_score(y_train,predict_train))


print("***********ADA booster****************")
dtc = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=25)
dtc.fit(x_train,y_train)
predict = dtc.predict(x_test)
predict_train = dtc.predict(x_train)
print("Testing accuracy: ",accuracy_score(y_test,predict))
print("Training Accuracy: ",accuracy_score(y_train,predict_train))










'''dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(dtree,class_names = X.columns,rounded = True,filled = True, fontsize=10)
plt.show()'''































































    




























