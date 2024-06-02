import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

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
    
    img = image.load_img(cropped_img_path, target_size=(299, 299))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    features = model.predict(img_data)
    return features
