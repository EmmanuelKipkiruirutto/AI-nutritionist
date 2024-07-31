import tkinter as tk
from tkinter import messagebox
import pickle
import pandas as pd

# Load the model
with open('nutrition_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the food data
data = pd.read_csv('food_data.csv')

# Ensure that the 'Food' column is in lowercase for consistent comparison
data['Food'] = data['Food'].str.strip().str.lower()

# Initialize Tkinter
root = tk.Tk()
root.title("Personal AI Nutritionist")
root.geometry("500x400")

# Define the function to fetch food information
def get_food_info():
    food_name = entry_food.get().strip().lower()  # Convert input to lowercase and strip whitespace
    
    # Check if the food is in the dataset
    if food_name not in data['Food'].values:
        messagebox.showerror("Error", "Food name not found.")
        return
    
    # Process the food name
    food_info = data[data['Food'] == food_name].iloc[0]
    
    # Predict using the model
    try:
        # Prepare the features for prediction
        features = food_info[['Calories', 'Protein', 'Fat', 'Carbohydrate', 'Fiber', 'Sugar']].values.reshape(1, -1)
        
        # Check if the model is correctly loaded and predict
        if hasattr(model, 'predict'):
            prediction = model.predict(features)
            
            # Ensure the prediction index is valid
            if 0 <= prediction[0] < len(data):
                predicted_food = data['Food'].iloc[prediction[0]]
                
                text_result.config(state=tk.NORMAL)
                text_result.delete(1.0, tk.END)
                text_result.insert(tk.END, f"Calories: {food_info['Calories']}\n")
                text_result.insert(tk.END, f"Protein: {food_info['Protein']}\n")
                text_result.insert(tk.END, f"Fat: {food_info['Fat']}\n")
                text_result.insert(tk.END, f"Carbohydrate: {food_info['Carbohydrate']}\n")
                text_result.insert(tk.END, f"Fiber: {food_info['Fiber']}\n")
                text_result.insert(tk.END, f"Sugar: {food_info['Sugar']}\n")
                text_result.config(state=tk.DISABLED)
            else:
                messagebox.showerror("Error", "Prediction index out of range.")
        else:
            messagebox.showerror("Error", "Model does not have a predict method.")
            
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create UI components
label_title = tk.Label(root, text="Personal AI Nutritionist", font=("Helvetica", 16))
label_title.pack(pady=10)

label_instruction = tk.Label(root, text="Enter the food name to get nutritional information:")
label_instruction.pack(pady=5)

entry_food = tk.Entry(root, width=50)
entry_food.pack(pady=5)

button_submit = tk.Button(root, text="Get Info", command=get_food_info)
button_submit.pack(pady=10)

text_result = tk.Text(root, height=10, width=60, wrap=tk.WORD, state=tk.DISABLED)
text_result.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
