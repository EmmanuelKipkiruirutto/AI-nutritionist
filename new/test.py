import pickle
import pandas as pd

# Load the model and food names
with open('nutrition_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('food_names.pkl', 'rb') as file:
    food_names = pickle.load(file)

# Define feature names based on training
feature_names = ['Calories', 'Protein', 'Fat', 'Carbohydrate', 'Fiber', 'Sugar']

# Create a DataFrame with the correct feature names
sample_input = pd.DataFrame([[250,26,15,0,0,0]], columns=feature_names)

# Predict
prediction = model.predict(sample_input)

# Map the prediction to the food name
predicted_index = prediction[0]
predicted_food = food_names[predicted_index]
print(f'Predicted Food: {predicted_food}')
