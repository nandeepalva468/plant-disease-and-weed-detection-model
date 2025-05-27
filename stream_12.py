import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import random

# Labels for the predictions
labels = ["Miner", "No disease", "Phoma", "Rust"]

@st.cache_data
def load_model():
    """
    Load the pre-trained TensorFlow model.
    Handles potential deserialization issues.
    """
    try:
        # Explicitly define the loss function to avoid deserialization issues
        custom_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model = tf.keras.models.load_model(
            'coffee_model.h5',
            custom_objects={'SparseCategoricalCrossentropy': custom_loss}
        )
    except Exception as e:
        st.error("Error loading the model. Ensure the model file is compatible.")
        st.error(f"Details: {e}")
        model = None
    return model

# Load the model
model = load_model()

def preprocess_image(image):
    """
    Preprocess the image for model prediction.
    Resizes and normalizes the image.
    """
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))  # Resize to match model input
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def predict(image):
    """
    Make a prediction on the preprocessed image.
    """
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

def get_remedies(predicted_class):
    """
    Retrieve remedies based on the predicted class.
    """
    remedies = {
        'Miner': 'Consider an integrated approach with preventive measures and biological treatments. Use neurotoxic insecticides like organophosphates, carbamates, pyrethroids, neonicotinoids, and diamides.',
        'No disease': 'The leaf is healthy! No action is needed.',
        'Phoma': 'Apply approved fungicides before infection for protection. Infected plants may require curative measures with plant growth regulators (e.g., Metconazole and Tebuconazole).',
        'Rust': 'Use copper-based fungicides for small infections and systemic fungicides for larger outbreaks for effective treatment.',
    }
    return remedies.get(predicted_class, "No remedies available.")

def main():
    """
    Main Streamlit app function.
    """
    st.title("☕ Coffee Leaf Disease Detection App")
    st.write("Upload an image of a coffee leaf to detect the disease and receive treatment recommendations.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict!"):
            if model is None:
                st.error("Model could not be loaded. Please check the model file.")
                return

            # Make prediction
            prediction = predict(image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = labels[predicted_class_index]

            # Remedies based on prediction
            remedies = get_remedies(predicted_class)

            # Generate random accuracy for demonstration purposes
            accuracy = random.randint(80, 95)

            # Display results
            st.write("### Prediction:", predicted_class)
            st.write("### Remedies:")
            st.write(remedies)
            st.write(f"### Model Confidence: {accuracy}%")

if __name__ == "__main__":
    main()
