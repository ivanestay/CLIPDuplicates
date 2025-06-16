import os
import time
import math
from PIL import Image
import torch
import streamlit as st
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
from collections import defaultdict

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
    div[data-testid="stTextInput"] {
        background-color: #540000;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    
    /* Containers (FileUploader, TextInput, etc.) */
    div[data-testid="stTextInput"] label {
        background-color: #540000;
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
    
    /* Style the File Uploader text */
    div[data-testid="stFileUploader"] label {
        color: white;
        transition: color 0.5s ease;
    }
    
    /* Style the "Browse files" button */
    [data-testid="stFileUploader"] button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease, color 0.3s ease;
        border: none;
    }
    
    /* Move the 'X' button (clear file) slightly to the right */
    [data-testid="stFileUploader"] section + div button {
        margin-left: 10px !important;  /* Increase space between filename and X button */
        transform: translateX(6px);     /* Optional fine-tuning */
    }
    
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
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
    
    /* --- File Uploader & Text Input Containers --- */
    div[data-testid="stTextInput"] {
        background-color: #ffb6b6;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    
    /* Containers (FileUploader, TextInput, etc.) */
    div[data-testid="stTextInput"] label {
        background-color: #ffb6b6;
        color: black;
        border-radius: 5px;
        padding: 10px;
        transition: background-color 0.5s ease, color 0.5s ease;
    }

    /* --- Input text color and background for TextInput --- */
    div[data-testid="stTextInput"] input {
        background-color: #ff4b4b !important;
        color: black !important;
        border: none;
        transition: background-color 0.5s ease, color 0.5s ease;
    }
    
    /* File uploader text */
    div[data-testid="stFileUploader"] label {
        color: black;
        transition: color 0.5s ease;
    }

    /* Radio button text (Light/Dark) */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] > div > div {
        color: black;
        transition: color 0.5s ease;
    }
    
    [data-testid="stFileUploader"] button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: background-color 0.5s ease, color 0.5s ease;
        border: none;
    }
    
    [data-testid="stFileUploader"] section + div button {
        margin-left: 10px !important;  /* Increase space between filename and X button */
        transform: translateX(6px);     /* Optional fine-tuning */
    }
    
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
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
st.title("🧠 CLIP-Based Duplicate & Similar Image Finder")
st.markdown("Upload a folder of images and find visually similar or duplicate groups.")

comparison_folder = st.text_input("📁 Enter Folder Path to Photos")
similarity_threshold = st.slider("🔗 Similarity Threshold (Higher = Stricter)", 0.80, 0.99, 0.90, step=0.01)

# --- File Uploader ---
uploaded_zip = st.file_uploader("📦 Upload a ZIP of Images", type=["zip"])
similarity_threshold = st.slider("🔗 Similarity Threshold (Higher = Stricter)", 0.80, 0.99, 0.90, step=0.01)

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "images.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        image_paths = [
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        if len(image_paths) < 2:
            st.warning("⚠️ Need at least two images to compare.")
        else:
            st.info(f"🔄 Loading and embedding {len(image_paths)} images...")
            image_tensors = []
            valid_paths = []

            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            for idx, path in enumerate(image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    image_tensors.append(preprocess(img))
                    valid_paths.append(path)
                except Exception as e:
                    st.error(f"Error loading {path}: {e}")

                progress_bar.progress((idx + 1) / len(image_paths))
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                eta = avg_time * (len(image_paths) - (idx + 1))
                mins, secs = divmod(int(eta), 60)
                status_text.markdown(
                    f"<p style='color:white if theme == \":red[Dark]\" else black;'>Estimated time remaining: {mins:02d}:{secs:02d}</p>",
                    unsafe_allow_html=True
                )

            batch_tensor = torch.stack(image_tensors).to(device)
            with torch.no_grad():
                image_embeddings = model.get_image_features(pixel_values=batch_tensor)
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            similarity_matrix = cosine_similarity(image_embeddings.cpu().numpy())

            st.success("✅ Similarity computed. Grouping similar images...")
            grouped = defaultdict(list)
            visited = set()
            N = len(valid_paths)

            for i in range(N):
                if i in visited:
                    continue
                group = [i]
                for j in range(i + 1, N):
                    if j not in visited and similarity_matrix[i][j] >= similarity_threshold:
                        group.append(j)
                if len(group) > 1:
                    for idx in group:
                        visited.add(idx)
                    grouped[i] = group

            if not grouped:
                st.warning("😕 No visually similar image groups found above the threshold.")
            else:
                st.subheader("🔍 Similar Image Groups")
                for group_id, indices in grouped.items():
                    with st.expander(f"Group {group_id + 1} ({len(indices)} images)"):
                        cols = st.columns(min(len(indices), 5))
                        for col, idx in zip(cols * (len(indices) // len(cols) + 1), indices):
                            with col:
                                st.image(Image.open(valid_paths[idx]),
                                         caption=os.path.basename(valid_paths[idx]),
                                         use_container_width=True)
else:
    st.info("📥 Please upload a ZIP file containing at least two images.")