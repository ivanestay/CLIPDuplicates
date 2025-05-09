import os
import time
import math
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import streamlit as st

# --- Load CLIP model and processor with caching ---
@st.cache_resource
def load_model_and_processor():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device

model, processor, device = load_model_and_processor()
#st.write(f"✅ Using device: `{device}`")

# --- Theme Toggle ---
theme = st.radio("💡 Select Theme", [":red[Light]", ":red[Dark]"], horizontal=True)

# --- Dynamic CSS Injection for Full App ---
if theme == ":red[Dark]":
    dark_theme = """
    <style>
    html, body, [class*="stApp"] {
        background-color: #0e1117;
        color: white;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    /* Change widgets (buttons, sliders, etc.) */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
    .stSlider label, .stSlider span {
        color: white;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    .stSlider>div>div {
        background: #0e1117;
        color: white;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    .stProgress>div>div>div {
        background-color: white;
        color: #4CAF50;
    }
    
    /* Containers (FileUploader, TextInput, etc.) */
    div[data-testid="stFileUploader"], div[data-testid="stTextInput"], 
    div[data-testid="stFileUploader"] label, div[data-testid="stTextInput"] label,
     div[data-testid="stFileUploader"] > div > div, div[data-testid="stTextInput"] > div > div {
        background-color: #222430;
        color: white;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* Radio button text (Light/Dark) */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] {
        color: white;
        transition: color 0.5s ease;
    }
    </style>
    """
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    light_theme = """
    <style>
    html, body, [class*="stApp"] {
        background-color: white;
        color: black;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    /* Change widgets (buttons, sliders, etc.) */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSlider label, .stSlider span {
        color: black;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    .stSlider>div>div {
        background: white;
        color: black;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    .stProgress>div>div>div {
        background-color: black;
        color: #4CAF50;
    }
    
    /* Containers (FileUploader, TextInput, etc.) */
    div[data-testid="stFileUploader"], div[data-testid="stTextInput"], 
    div[data-testid="stFileUploader"] label, div[data-testid="stTextInput"] label,
     div[data-testid="stFileUploader"] > div > div, div[data-testid="stTextInput"] > div > div {
        background-color: #f5f5f5;
        color: black;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* Radio button text (Light/Dark) */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] > div > div {
        color: black;
        transition: color 0.5s ease;
    }
    </style>
    """
    st.markdown(light_theme, unsafe_allow_html=True)

# --- Similarity function ---
def compute_similarity(image_path, ref_embedding):
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        similarity = (ref_embedding @ embedding.T).item()
        return similarity
    except Exception as e:
        st.error(f"Error with {image_path}: {e}")
        return None

# --- Streamlit UI ---
st.title("🔍 CLIP-Based Photo Similarity Finder")
st.markdown("Upload a reference image and select a folder of photos to compare against it.")

reference_image = st.file_uploader("📌 Upload a Reference Image", type=["jpg", "jpeg", "png"])
comparison_folder = st.text_input("📁 Enter Folder Path to Photos")

top_k = st.slider("🔢 Number of Top Matches to Display", min_value=1, max_value=20, value=5)

if reference_image and comparison_folder:
    if not os.path.isdir(comparison_folder):
        st.error("❌ The specified comparison folder path is invalid or does not exist.")
    else:
        # Load reference image and get embedding
        ref_img = Image.open(reference_image).convert("RGB")
        st.image(ref_img, caption="Reference Image", use_container_width=True)

        ref_inputs = processor(images=ref_img, return_tensors="pt").to(device)
        with torch.no_grad():
            ref_embedding = model.get_image_features(**ref_inputs)
        ref_embedding = torch.nn.functional.normalize(ref_embedding, p=2, dim=1)

        # Get list of images
        image_paths = [os.path.join(comparison_folder, f)
                       for f in os.listdir(comparison_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(image_paths) == 0:
            st.warning("⚠️ No images found in the specified folder.")
        else:
            st.markdown("### 🖼️ Top Matches")
            results = []

            # Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()
            total_images = len(image_paths)

            for idx, filepath in enumerate(image_paths):
                sim = compute_similarity(filepath, ref_embedding)
                if sim is not None:
                    results.append((filepath, sim))

                # Update progress
                progress_percent = (idx + 1) / total_images
                progress_bar.progress(progress_percent)

                # Update estimated time left
                elapsed = time.time() - start_time
                if idx > 0:
                    avg_time_per_image = elapsed / (idx + 1)
                    estimated_remaining = avg_time_per_image * (total_images - idx - 1)

                    minutes = math.floor(estimated_remaining / 60)
                    seconds = int(estimated_remaining % 60)
                    status_text.text(f"Progress: {progress_percent:.0%} | Estimated time left: {minutes}m {seconds}s")

            # Sort results and display top matches
            results.sort(key=lambda x: x[1], reverse=True)
            top_results = results[:top_k]

            for img_path, score in top_results:
                img = Image.open(img_path)
                st.image(img, caption=f"{os.path.basename(img_path)} (Similarity: {score:.2f})", use_container_width=True)

else:
    st.info("📥 Please upload a reference image and provide a valid folder path.")