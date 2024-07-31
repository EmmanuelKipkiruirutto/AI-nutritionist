import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = pd.read_csv('food_data.csv')
data['Food'] = data['Food'].astype('category')
data['FoodCodes'] = data['Food'].cat.codes

# Define features and target
X = data[['Calories', 'Protein', 'Fat', 'Carbohydrate', 'Fiber', 'Sugar']]
y = data['FoodCodes']

# Save food names mapping
food_names = data['Food'].cat.categories

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and food names
with open('nutrition_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('food_names.pkl', 'wb') as file:
    pickle.dump(food_names, file)
