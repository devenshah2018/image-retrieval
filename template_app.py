import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Image Retrieval App Demo", layout="wide")

# Centered title
st.markdown("<h1 style='text-align: center;'>Image Retrieval App Demo</h1>", unsafe_allow_html=True)

# Upload and action buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

with col2:
    if st.button("Search"):
        st.write("Search initiated")

with col3:
    if st.button("Clear"):
        st.write("Clear initiated")

st.markdown('---')

# Display results
st.header("Results")
cols = st.columns(3)

# Placeholder images for layout demonstration
placeholder_image_urls = [
    "https://picsum.photos/id/237/300/200",
    "https://picsum.photos/id/238/300/200",
    "https://picsum.photos/id/239/300/200"
]

for i, col in enumerate(cols):
    with col:
        try:
            response = requests.get(placeholder_image_urls[i])
            image = Image.open(BytesIO(response.content))
            st.image(image, caption=f"Result {i+1}")
        except Exception as e:
            st.write(f"Error loading image {i+1}: {e}")

if uploaded_file:
    st.sidebar.header("Uploaded Image")
    image = Image.open(uploaded_file)
    st.sidebar.image(image, use_column_width=True)
