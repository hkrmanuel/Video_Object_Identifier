import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Load the InceptionV3 model
model = InceptionV3(weights='imagenet')

# Function to preprocess image for InceptionV3
def preprocess_frame(frame_path):
    img = image.load_img(frame_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)
    return x

# Get predictions for each frame
def get_predictions(frame_path):
    x = preprocess_frame(frame_path)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

# Extract frames from the video using your specific method
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success, image = vidcap.read()
    while success:
        frame_path = os.path.join(output_folder, f"frame{count}.jpg")
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1
    return count

# Function to search for an object in the frames
def search_for_object(frames_with_objects, search_query):
    results = []
    for frame_path, predictions in frames_with_objects.items():
        for _, label, _ in predictions:
            if search_query in label.lower():
                results.append(frame_path)
                break
    return results

# Streamlit app
st.title("Video Object Predictor")
st.write("Upload a video file (max 10MB) to extract frames")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
memory_size_threshold = 10 * 1024 * 1024  # 10 MB

if uploaded_file is not None:
    file_size = uploaded_file.size
    if file_size > memory_size_threshold:
        st.error(f"The file exceeds the memory size threshold of {memory_size_threshold / (1024 * 1024)} MB.")
    else:
        video_path = os.path.join("temp_video", uploaded_file.name)
        os.makedirs("temp_video", exist_ok=True)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("The file is within the memory size threshold.")
        output_folder = 'frames'
        frames_with_objects = {}

        frame_count = extract_frames(video_path, output_folder)
        st.success(f'Extracted {frame_count} frames from the video.')
        
        for i in tqdm(range(frame_count)):
            frame_path = os.path.join(output_folder, f"frame{i}.jpg")
            predictions = get_predictions(frame_path)
            frames_with_objects[frame_path] = predictions

        search_query = st.text_input("Enter the object you want to search for: ").strip().lower()

        if search_query:
            search_results = search_for_object(frames_with_objects, search_query)
            if search_results:
                st.write(f'Found {search_query} in the following frames:')
                for result in search_results:
                    st.image(result, use_column_width=True)
            else:
                st.error("Object doesn't exist!!!")

        # Clean up
        os.remove(video_path)
        for frame_file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, frame_file))
        os.rmdir(output_folder)