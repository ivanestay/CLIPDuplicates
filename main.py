import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import streamlit as st

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Checks if a GPU is available to speed up CLIP similarity
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval()

# Similarity function
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
        print(f"Error with {image_path}: {e}")
        return None

# Streamlit UI
st.title("CLIP-Based Photo Similarity")

st.markdown("Upload a reference image and select a folder of photos to compare.")

reference_image = st.file_uploader("📌 Upload Reference Image", type=["jpg", "jpeg", "png"])
comparison_folder = st.text_input("📁 Enter Folder Path for Photos")

if reference_image and comparison_folder and os.path.exists(comparison_folder):
    # Load reference image and get embedding
    ref_img = Image.open(reference_image).convert("RGB")
    st.image(ref_img, caption="Reference Image", use_container_width=True)

    ref_inputs = processor(images=ref_img, return_tensors="pt").to(device)
    with torch.no_grad():
        ref_embedding = model.get_image_features(**ref_inputs)
    ref_embedding = torch.nn.functional.normalize(ref_embedding, p=2, dim=1)

    st.markdown("### 🔍 Top Matches")
    results = []
    image_paths = [f for f in os.listdir(comparison_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    progress = st.progress(0)
    for idx, filename in enumerate(image_paths):
        filepath = os.path.join(comparison_folder, filename)
        sim = compute_similarity(filepath, ref_embedding)
        if sim is not None:
            results.append((filename, sim))
        progress.progress((idx + 1) / len(image_paths))

    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:5]

    for fname, score in top_results:
        img_path = os.path.join(comparison_folder, fname)
        img = Image.open(img_path)
        st.image(img, caption=f"{fname} (Similarity: {score:.2f})", use_container_width=True)

elif reference_image and not os.path.exists(comparison_folder):
    st.error("The specified comparison folder path does not exist.")