import pandas as pd
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.tree     import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


df = pd.read_csv("../../data/playNoPlay.csv")
#print(df)
#print(df.info())
#for col in df.columns:
    #print(df[col].value_counts()) 
x= df[['outlook','temp','humidity','windy']] #
y = df['play']
le= LabelEncoder()
y= le.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)
encoder = ce.OrdinalEncoder(cols=['outlook','temp','humidity','windy'])
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

obj = DecisionTreeClassifier(criterion="entropy",  max_depth=3, random_state=0)
dtree = obj.fit(x_train,y_train)
y_train_pred = obj.predict(x_train)
y_test_pred = obj.predict(x_test)

plt.figure(figsize=(50,30))
tree.plot_tree(dtree, class_names=x.columns)
plt.show()
print("************Decision Tree Classifier***************")
print("Testing accuracy: ",accuracy_score(y_test,y_test_pred))
print("Training Accuracy: ",accuracy_score(y_train,y_train_pred))

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

print("************Gradient Boosting Classifier***************")
RFC = GradientBoostingClassifier(n_estimators=100)
RFC.fit(x_train,y_train)
predict = RFC.predict(x_test)
predict_train = dtc.predict(x_train)
print("Testing accuracy: ",accuracy_score(y_test,predict))
print("Training Accuracy: ",accuracy_score(y_train,predict_train))
print(classification_report(y_test,predict))

'''
LabelEncoder should be used to encode target values, 
i.e. y, and not the input X. Ordinal encoding should be 
used for ordinal variables (where order matters,
like cold , warm , hot ); vs Label encoding should be 
used for non-ordinal (aka nominal) variables

'''