import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess the dataset for the text-based model
@st.cache_resource
def load_text_model_data():
    # Replace with your actual dataset path
    data = pd.read_csv("dataset.csv")
    le = LabelEncoder()
    data["weather_encoded"] = le.fit_transform(data["weather"])
    X = data[["precipitation", "temp_max", "temp_min", "wind"]]
    y = data["weather_encoded"]

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le

@st.cache_resource
def load_text_model():
    X_train, _, y_train, _, _ = load_text_model_data()
    model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20, class_weight="balanced", random_state=0)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def load_vision_model():
    # Replace with the path to your trained vision model
    return load_model("wather.h5")

# Load models
text_model = load_text_model()
vision_model = load_vision_model()
_, _, _, _, label_encoder = load_text_model_data()

# Streamlit UI
st.title("Weather Prediction App")
st.sidebar.title("Choose Prediction Type")

# Sidebar options
prediction_type = st.sidebar.radio(
    "Select the type of prediction:",
    ("Text-Based Prediction", "Image-Based Prediction")
)

if prediction_type == "Text-Based Prediction":
    st.header("Text-Based Weather Prediction")
    st.write("Enter the weather parameters below:")

    # Input fields for weather parameters
    precipitation = st.number_input("Precipitation (mm)", min_value=0.0, step=0.1)
    temp_max = st.number_input("Max Temperature (°C)", step=0.1)
    temp_min = st.number_input("Min Temperature (°C)", step=0.1)
    wind = st.number_input("Wind Speed (m/s)", min_value=0.0, step=0.1)

    # Predict button
    if st.button("Predict Weather"):
        input_data = np.array([[precipitation, temp_max, temp_min, wind]])
        prediction = text_model.predict(input_data)[0]
        weather_map = {index: label for index, label in enumerate(label_encoder.classes_)}
        st.write(f"Predicted Class Index: {prediction}")  # Debugging: Show predicted index
        st.success(f"The predicted weather is: {weather_map.get(prediction, 'Unknown')}")

elif prediction_type == "Image-Based Prediction":
    st.header("Image-Based Weather Prediction")
    st.write("Upload a satellite image to predict the weather:")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and preprocess the image
        test_img = load_img(uploaded_file, target_size=(224, 224))
        test_img_array = img_to_array(test_img) / 255.0  # Normalize pixel values
        test_img_array = np.expand_dims(test_img_array, axis=0)  # Add batch dimension

        # Predict button
        if st.button("Predict Weather"):
            predicted_prob = vision_model.predict(test_img_array)[0][0]
            #st.write(f"Predicted Probability (Rainy): {predicted_prob:.2f}")  # Debugging: Show probability

            # Adjust threshold dynamically
            threshold = 0.5  # Default threshold
            if predicted_prob < threshold:
                predicted_class = "Rainy"
            else:
                predicted_class = "Sunny"

            # Debugging: Show threshold and classification decision
            #st.write(f"Threshold: {threshold}")
            st.write(f"Classification Decision: {predicted_class}")

            st.image(test_img, caption=f"Predicted: {predicted_class}", use_column_width=True)
            st.success(f"The predicted weather is: {predicted_class}")


# Example inputs for testing the text-based model

#Weather	Precipitation (mm)	Temp Max (°C)	Temp Min (°C)	Wind Speed (m/s)
#Drizzle	0.0	                  10.0	             8.0           	0.5
#Rain	    10.0	              22.0	             18.0	        5.0
#Snow	    5.0	                  -2.0	             -5.0	        3.0
#Sun	    0.0	                  30.0	             20.0	        1.0
#fog        0.0                   8.0                 6.0           0.2


# Here’s the weather update for Bogotá today:

# Precipitation: 0.17 in (4.3 mm)

#M ax Temperature: 57°F (14°C)

# Min Temperature: 50°F (10°C)

# Wind Speed: Data not available
#-------------------------------------------------------------------------------
# Here’s the weather update for Nagpur today:

# Precipitation: 0.00 mm

# Max Temperature: 111°F (44°C)

# Min Temperature: 89°F (32°C)

# Wind Speed: 0.00 m/s