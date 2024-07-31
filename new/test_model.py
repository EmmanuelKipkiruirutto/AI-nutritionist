import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
data = pd.read_csv('food_data.csv')

# Prepare features and target
X = data[['Calories', 'Protein', 'Fat', 'Carbohydrate', 'Fiber', 'Sugar']]
y = data['Food']  # Assuming the target is the food name

# Encode target labels
y = y.astype('category').cat.codes

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model
model = joblib.load('nutrition_model.pkl')

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")
