import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pathlib 

model = tf.keras.models.load_model('full_model.h5')


data_dir = pathlib.Path(r"C:\Users\PAVAN KALYAN\OneDrive\Documents\IV-II\Major Project\Selected\Bird Species\bird dataset\train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

def preprocess_image(image):
    img = image.resize((300, 300))  
    img_array = np.array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Function to make predictions
def predict_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return class_names[predicted_class], confidence

def main():
    st.title("Bird Species Classification")
    st.write("Upload an image of a bird to classify its species.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        class_name, confidence = predict_image(image)
        st.write("Class:", class_name)
        st.write("Confidence:", confidence)

if __name__ == "__main__":
    main()
