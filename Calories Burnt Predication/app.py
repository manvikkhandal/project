import streamlit as st
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd
from PIL import Image

# Load the trained model
with open('Model/decision_tree_regressor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Database connection
def init_db():
    conn = sqlite3.connect('Database/Calorie.db')
    cursor = conn.cursor()
    return conn, cursor

# Insert new entry into the database
def insert_calorie_prediction(cursor, calories, gender, age, height, weight, duration, heart_rate, body_temp):
    date = datetime.now().strftime('%Y-%m-%d')
    day = datetime.now().strftime('%A')
    cursor.execute('''
    INSERT INTO calorie_predictions (date, day, calories, gender, age, height, weight, duration, heart_rate, body_temp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (date, day, calories, gender, age, height, weight, duration, heart_rate, body_temp))

    # Delete oldest entries if there are more than 7 rows
    cursor.execute('SELECT COUNT(*) FROM calorie_predictions')
    row_count = cursor.fetchone()[0]
    if row_count > 7:
        cursor.execute('''
        DELETE FROM calorie_predictions 
        WHERE id IN (SELECT id FROM calorie_predictions ORDER BY id ASC LIMIT ?)
        ''', (row_count - 7,))

# Fetch the latest 7 entries from the database
def get_latest_entries(cursor):
    cursor.execute('''
    SELECT date, day, calories, gender, age, height, weight, duration, heart_rate, body_temp
    FROM calorie_predictions
    ORDER BY id DESC LIMIT 7
    ''')
    return cursor.fetchall()

# Initialize the database
conn, cursor = init_db()

# Title of the app, centered with inline style
st.markdown('<h1 style="text-align: center;">Calorie Burn Prediction</h1>', unsafe_allow_html=True)

# Load and display an image in the center
image = Image.open('Assets/calorie.jpg')  # Make sure to update with your image path
resized_image = image.resize((700, 250))  # Resize image to 250x250 pixels
st.image(resized_image, use_column_width=False, caption='Predict your Calorie Burn', output_format='PNG')

# Input fields for the user to provide input data, wrapped in centered subheading
st.markdown('<h2 style="text-align: center;">Enter Your Details</h2>', unsafe_allow_html=True)

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, step=1)
height = st.number_input("Height (in cm)", min_value=100, max_value=250, step=1)
weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, step=1)
duration = st.number_input("Duration of Exercise (in minutes)", min_value=1, max_value=500, step=1)
heart_rate = st.number_input("Heart Rate (in bpm)", min_value=60, max_value=200, step=1)
body_temp = st.number_input("Body Temperature (in °C)", step=0.1)  # No limits for Body Temp

# Convert categorical 'Gender' to numerical values
gender_numeric = 1 if gender == "Male" else 0

# Prepare the input features
input_data = np.array([[gender_numeric, age, height, weight, duration, heart_rate, body_temp]])

# Centered prediction button
if st.button("Predict Calorie Burn"):
    # Make prediction
    prediction = model.predict(input_data)
    calories_burned = prediction[0]

    # Display the result in the center
    st.markdown(f'<h3 style="text-align: center;">Estimated Calorie Burn: {calories_burned:.2f} calories</h3>', unsafe_allow_html=True)

    # Insert prediction data into the database and manage entries
    insert_calorie_prediction(cursor, calories_burned, gender, age, height, weight, duration, heart_rate, body_temp)
    
    # Commit the changes
    conn.commit()

# Display the latest 7 entries, wrapped in centered subheading
st.markdown('<h2 style="text-align: center;">Latest Entries</h2>', unsafe_allow_html=True)

entries = get_latest_entries(cursor)

if entries:
    # Convert entries to a DataFrame
    df = pd.DataFrame(entries, columns=["Date", "Day", "Calories", "Gender", "Age", "Height", "Weight", "Duration", "Heart Rate", "Body Temp"])
    
    # Style the table to center align the content
    st.markdown("""
    <style>
    .center-table th, .center-table td {
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the table
    st.write('<div class="center-table">', unsafe_allow_html=True)
    st.dataframe(df)
    st.write('</div>', unsafe_allow_html=True)
else:
    st.write("No entries found.")

# Close the database connection when the app ends
conn.close()