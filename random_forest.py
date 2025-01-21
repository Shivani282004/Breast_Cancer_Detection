import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('Breast Cancer Data.csv')  # Ensure this file path is correct in your local setup

# Remove columns where the column name contains 'Unnamed'
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Check for NaN and Infinite values in the dataset
print("Checking for NaN values:")
print(data.isnull().sum())  # Check for NaN values in each column

# Handle NaN values by filling with the mean of each column
data = data.fillna(data.mean())  # Filling NaN values with the mean of the respective column

# Handle Infinite values by replacing with NaN and then filling with mean
data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
data = data.fillna(data.mean())

# Preprocess the dataset
X = data.drop(columns=['diagnosis'])  # Features
y = data['diagnosis']  # Target

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode the target variable (diagnosis)
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model as a .joblib file
joblib.dump(model, 'cancer_model.joblib')
print("Model saved as 'cancer_model.joblib'")
