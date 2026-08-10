from decision_tree import MyTree

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = load_breast_cancer()
X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.2,random_state=42)

model = MyTree()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
print("Accuracy:",model.calculateAcc(Y_test,y_pred))

print(classification_report(y_pred,Y_test))