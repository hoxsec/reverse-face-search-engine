import os
import uuid
import streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from database import store_features, create_table, get_all_features, clear_database
from search import search_similar_images  # Importing the search function

# Ensure the images directories exist
inserted_images_dir = 'images/inserted'
processed_images_dir = 'images/processed'
os.makedirs(inserted_images_dir, exist_ok=True)
os.makedirs(processed_images_dir, exist_ok=True)

# Load pre-trained model for face embeddings
model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    # Assuming the first detected face is the one we want to crop
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    cropped_img_path = img_path.replace(".jpg", "_cropped.jpg")
    cv2.imwrite(cropped_img_path, face)
    
    return cropped_img_path

def extract_features(img_path):
    cropped_img_path = detect_and_crop_face(img_path)
    
    if cropped_img_path is None:
        return None
    
    img = keras_image.load_img(cropped_img_path, target_size=(299, 299))
    img_data = keras_image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    features = model.predict(img_data)
    return features

def is_duplicate(new_features, threshold=0.5, top_k=10):
    # Get the top_k similar images
    top_matches = search_similar_images(new_features, top_k=top_k)
    
    for dist in top_matches:
        if dist < threshold:
            return True
    
    return False

def insert_image(img):
    if img is None:
        return "No image uploaded"
    
    # Generate a unique filename
    img_id = str(uuid.uuid4())
    img_path = f"{inserted_images_dir}/{img_id}.jpg"
    
    # Save the image
    img.save(img_path)
    
    # Extract features of the uploaded image
    features = extract_features(img_path)
    
    if features is None:
        return "No face detected in the uploaded image"
    
    # Check for duplicates
    if is_duplicate(features):
        return "Duplicate image detected. Upload a different image."
    
    # Store the features in the database
    store_features(img_path, features)
    
    return f"Image inserted into the database with ID: {img_id}"

def process_image(img):
    if img is None:
        return "No image uploaded"
    
    # Generate a unique filename for the query image
    img_id = str(uuid.uuid4())
    img_path = f"{processed_images_dir}/{img_id}.jpg"
    
    # Save the image
    img.save(img_path)
    
    # Extract features of the uploaded image
    query_features = extract_features(img_path)
    
    if query_features is None:
        return "No face detected in the uploaded image"
    
    # Search for the top 10 similar images
    top_matches = search_similar_images(query_features, top_k=10)
    
    if not top_matches:
        return "No matching images found"
    
    # Load the top match images
    top_match_images = [Image.open(match[1]) for match in top_matches]
    
    return top_match_images

def clear_cache():
    # Clear the database
    clear_database()
    
    # Remove all files in inserted and processed directories
    for folder in [inserted_images_dir, processed_images_dir]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    return "Cache cleared successfully"

# Streamlit UI
st.title("Face Image Search Engine")

# Tabs
tab1, tab2, tab3 = st.tabs(["Insert Image", "Search Image", "Clear Cache"])

with tab1:
    st.header("Insert Image")
    uploaded_image = st.file_uploader("Upload an image to insert into the database.", type=["jpg", "jpeg", "png"])
    if st.button("Insert"):
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            result = insert_image(image)
            st.write(result)

with tab2:
    st.header("Search Image")
    uploaded_image = st.file_uploader("Upload an image to search for similar images.", type=["jpg", "jpeg", "png"], key="search")
    if st.button("Search"):
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            top_matches = process_image(image)
            if isinstance(top_matches, str):
                st.write(top_matches)
            else:
                st.image(top_matches, caption=['Match']*len(top_matches), use_column_width=True)

with tab3:
    st.header("Clear Cache")
    if st.button("Clear"):
        result = clear_cache()
        st.write(result)

if __name__ == "__main__":
    # Ensure the table exists
    create_table()
