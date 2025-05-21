import os
import time
import math
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import streamlit as st
from torchvision import transforms

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

    /* --- Buttons --- */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* --- Sliders --- */
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
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* --- File Uploader & Text Input Containers --- */
    div[data-testid="stFileUploader"], div[data-testid="stTextInput"] {
        background-color: #222430;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* --- Inner File Uploader box --- */
    div[data-testid="stFileUploader"] > div {
        background-color: #1c1e26 !important;
        border: 1px solid #3a3f4b !important;
        color: white;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    div[data-testid="stFileUploader"] input[type="file"] {
        color: white !important;
    }
    
    /* Containers (FileUploader, TextInput, etc.) */
    div[data-testid="stFileUploader"] label, div[data-testid="stTextInput"] label {
        background-color: #222430;
        color: white;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* --- Input text color and background for TextInput --- */
    div[data-testid="stTextInput"] input {
        background-color: #1c1e26 !important;
        color: white !important;
        border: none;
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* --- Radio button text --- */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] {
        color: white;
        transition: background-color 0.5s ease, color 0.5s ease;
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

# --- Preprocessing for Batch ---
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

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
        # Load reference image and compute embedding
        ref_img = Image.open(reference_image).convert("RGB")
        st.image(ref_img, caption="Reference Image", use_container_width=True)

        ref_inputs = processor(images=ref_img, return_tensors="pt").to(device)
        with torch.no_grad():
            ref_embedding = model.get_image_features(**ref_inputs)
        ref_embedding = torch.nn.functional.normalize(ref_embedding, p=2, dim=1)

        # Gather image paths
        image_paths = [os.path.join(comparison_folder, f)
                       for f in os.listdir(comparison_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(image_paths) == 0:
            st.warning("⚠️ No images found in the specified folder.")
        else:
            st.markdown("### 🖼️ Top Matches")

            image_tensors = []
            valid_paths = []

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            total_images = len(image_paths)

            for idx, path in enumerate(image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    image_tensors.append(preprocess(img))
                    valid_paths.append(path)
                except Exception as e:
                    st.error(f"Error loading {path}: {e}")

                # Progress update
                progress_percent = (idx + 1) / total_images
                progress_bar.progress(progress_percent)
                elapsed = time.time() - start_time
                if idx > 0:
                    avg_time = elapsed / (idx + 1)
                    eta = avg_time * (total_images - idx - 1)
                    status_text.text(f"Progress: {progress_percent:.0%} | Estimated time left: {int(eta//60)}m {int(eta%60)}s")

            if len(image_tensors) == 0:
                st.warning("⚠️ No valid images could be loaded.")
            else:
                # Stack and move to device
                batch_tensor = torch.stack(image_tensors).to(device)

                with torch.no_grad():
                    image_embeddings = model.get_image_features(pixel_values=batch_tensor)
                image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)

                # Compute similarity in batch
                similarities = (ref_embedding @ image_embeddings.T).squeeze()
                results = list(zip(valid_paths, similarities.tolist()))

                # Show top results
                results.sort(key=lambda x: x[1], reverse=True)
                for img_path, score in results[:top_k]:
                    st.image(Image.open(img_path), caption=f"{os.path.basename(img_path)} (Similarity: {score:.2f})", use_container_width=True)
else:
    st.info("📥 Please upload a reference image and provide a valid folder path.")