import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import *
import os

# Page configuration
st.set_page_config(
    page_title="Image Histogram Equalizer",
    page_icon="🖼️",
    layout="wide"
)

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("### Image Source")
source_option = st.sidebar.radio(
    "Select image source:",
    ("Upload", "Sample Images")
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Main content
st.title("Image Histogram Equalization")
st.markdown("""
This application demonstrates different histogram equalization techniques for color images.
""")

# Image selection
image = None
if source_option == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
else:
    sample_images = os.listdir("src/images")
    selected_image = st.selectbox("Select a sample image", sample_images)
    if selected_image:
        image_path = os.path.join("src/images", selected_image)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Process and display results
if image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Display original histogram
        fig, ax = plt.subplots()
        ax.hist(image.ravel(), 256, [0, 256])
        ax.set_title('Original Histogram')
        ax.set_xlim([0, 256])
        st.pyplot(fig)
    
    # Processing methods
    method = st.radio(
        "Select equalization method:",
        ("RGB Channel-wise", "HSV V-channel", "CLAHE (LAB)")
    )
    
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            if method == "RGB Channel-wise":
                processed_image = channel_wise_hist_eq(image)
            elif method == "HSV V-channel":
                processed_image = hsv_hist_eq(image)
            else:
                processed_image = clahe_hist_eq(image)
            
            # Save processed image
            save_path = save_image(processed_image, "src/processed", "processed.jpg")
            
            # Update session state
            st.session_state.processed = True
            st.session_state.processed_image = processed_image
            st.session_state.method = method
    
    if st.session_state.processed:
        with col2:
            st.image(
                st.session_state.processed_image,
                caption=f"{st.session_state.method} Equalized",
                use_column_width=True
            )
            
            # Display processed histogram
            fig, ax = plt.subplots()
            ax.hist(st.session_state.processed_image.ravel(), 256, [0, 256])
            ax.set_title(f'{st.session_state.method} Equalized Histogram')
            ax.set_xlim([0, 256])
            st.pyplot(fig)
            
            # Download button
            with open("src/processed/processed.jpg", "rb") as file:
                st.download_button(
                    label="Download Processed Image",
                    data=file,
                    file_name=f"equalized_{st.session_state.method.lower().replace(' ', '_')}.jpg",
                    mime="image/jpeg"
                )
else:
    st.info("Please upload an image or select a sample image to get started.")