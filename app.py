

import streamlit as st
import openai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time

# Set up OpenAI API key
openai.api_key = "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"

# Load the model
model = load_model('weights.h5')

# Rate limit variables
max_requests = 60  # Maximum number of requests allowed
max_tokens = 150000  # Maximum tokens allowed
current_requests = 0  # Current number of requests made
current_tokens = 0  # Current number of tokens used
request_interval = 1  # Minimum interval between requests in seconds
last_request_time = 0

def preprocess_image(image):
    img = image.resize((128, 128))  # Resize the image
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function for OpenAI chatbot response with rate limiting
def get_chatbot_response(user_input):
    global current_requests, current_tokens, last_request_time

    current_time = time.time()
    
    # Wait if the minimum interval hasn't passed
    if current_time - last_request_time < request_interval:
        time.sleep(request_interval - (current_time - last_request_time))
    
    # Check if we can make a new request
    if current_requests >= max_requests:
        st.warning("Rate limit exceeded for requests. Please wait for the next reset.")
        return "Rate limit exceeded for requests."
    
    messages = [
        {"role": "user", "content": user_input}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=messages
        )

        # Update request count and token usage
        current_requests += 1
        tokens_used = response['usage']['total_tokens']  # Get the number of tokens used in the response
        current_tokens += tokens_used

        # Check token limits
        if current_tokens > max_tokens:
            st.warning("Rate limit exceeded for tokens. Please wait for the next reset.")
            return "Rate limit exceeded for tokens."

        last_request_time = time.time()  # Update the last request time
        return response['choices'][0]['message']['content']
    
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return "I'm sorry, I'm having trouble processing your request right now."

# Streamlit app layout
st.title("Image Classification with TensorFlow and OpenAI Chatbot")
st.write("Upload an image to classify it as malignant or not.")

# Chatbot Interface
st.subheader("Chatbot")
user_input = st.text_input("You:", "")

if user_input:
    response = get_chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=None)

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)
    class_label = "Malignant" if prediction[0] > 0.5 else "Not Malignant"  # Threshold for binary classification

    # Display the result
    st.write(f"Prediction: {class_label}")
