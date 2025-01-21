import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Breast Cancer Data.csv')  


data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


print("Checking for NaN values:")
print(data.isnull().sum())  

data = data.fillna(data.mean())  

data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
data = data.fillna(data.mean())

X = data.drop(columns=['diagnosis']) 
y = data['diagnosis']  # Target


label_encoder = LabelEncoder()


y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'cancer_model.joblib')
print("Model saved as 'cancer_model.joblib'")
