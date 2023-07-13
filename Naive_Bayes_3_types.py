import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import category_encoders as ce

df = pd.read_csv("../../data/playNoPlay.csv")
x= df[['outlook','temp','humidity','windy']] #
y = df['play']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
encoder = ce.OrdinalEncoder(cols=['outlook','temp','humidity','windy'])
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)



gnb = GaussianNB()
gnbfit=gnb.fit(x_train, y_train)
y_train_pred = gnb.predict(x_train)
y_test_pred = gnb.predict(x_test)
print("Testing accuracy: ",accuracy_score(y_test,y_test_pred))
print("Training Accuracy: ",accuracy_score(y_train,y_train_pred))
print(classification_report(y_test,y_test_pred))


mnb = MultinomialNB()
mnbfit=mnb.fit(x_train, y_train)
y_train_pred = mnb.predict(x_train)
y_test_pred = mnb.predict(x_test)
print("Testing accuracy: ",accuracy_score(y_test,y_test_pred))
print("Training Accuracy: ",accuracy_score(y_train,y_train_pred))
print(classification_report(y_test,y_test_pred))

bnb = BernoulliNB()
bnbfit=bnb.fit(x_train, y_train)
y_train_pred = bnb.predict(x_train)
y_test_pred = bnb.predict(x_test)
print("Testing accuracy: ",accuracy_score(y_test,y_test_pred))
print("Training Accuracy: ",accuracy_score(y_train,y_train_pred))
print(classification_report(y_test,y_test_pred))
