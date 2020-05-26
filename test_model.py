import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

dataset=pd.read_csv('Social_Network_Ads.csv')
X_train, X_test, y_train, y_test = train_test_split(dataset[['Age','EstimatedSalary']].values, dataset['Purchased'].values, test_size=0.33, random_state=42)
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

#SHYP1
model=KNeighborsClassifier(n_neighbors=1,)
#EHYP1

model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
