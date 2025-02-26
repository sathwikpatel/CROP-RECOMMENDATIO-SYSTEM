import streamlit as st
import numpy as np
import pickle

# Load the trained model and scalers
model = pickle.load(open(r'C:\Users\acer\Downloads\Crop-Recommendation-system\Crop-Recommendation-system\Crop_Recommendation-main', 'rb'))
sc = pickle.load(open(r'C:\Users\acer\Downloads\Crop-Recommendation-system\Crop-Recommendation-system\Crop_Recommendation-main', 'rb'))
mx = pickle.load(open(r'C:\Users\acer\Downloads\Crop-Recommendation-system\Crop-Recommendation-system\Crop_Recommendation-main', 'rb'))

# Crop dictionary mapping
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Starfruit", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Streamlit UI
st.title("🌾 Crop Recommendation System")
st.write("Enter the required parameters to predict the best crop to cultivate.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=500.0, step=0.1)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=500.0, step=0.1)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=500.0, step=0.1)
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

# Prediction logic
if st.button("Predict Crop"):
    feature_list = np.array([N, P, K, temp, humidity, ph, rainfall]).reshape(1, -1)
    
    mx_features = mx.transform(feature_list)
    sc_mx_features = sc.transform(mx_features)
    
    prediction = model.predict(sc_mx_features)[0]  # Get the predicted class

    if prediction in crop_dict: 
        st.success(f"✅ Recommended Crop: **{crop_dict[prediction]}**")
    else:
        st.error("❌ Sorry, we could not determine the best crop with the provided data.")
