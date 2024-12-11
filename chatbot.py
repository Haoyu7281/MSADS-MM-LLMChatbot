import streamlit as st
import faiss
import pickle
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

# Initialize Streamlit page configuration
st.set_page_config(page_title="Multimodal Product Search", layout="wide")

# Load the CLIP model and processor
@st.cache_resource
def load_clip_model():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

# Load FAISS index and metadata
@st.cache_resource
def load_faiss_and_metadata():
    faiss_index = faiss.read_index("faiss_index_short.bin")
    with open("metadata_short.pkl", "rb") as f:
        metadata = pickle.load(f)
    return faiss_index, metadata

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate query embedding
def generate_query_embedding(
    text_query: str = None,
    image: Image = None,
    text_weight: float = 0.5,
    image_weight: float = 0.5
) -> np.ndarray:
    text_embedding = None
    image_embedding = None

    # Generate text embedding
    if text_query:
        inputs = clip_processor(text=[text_query], return_tensors="pt").to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs).cpu().numpy()

    # Generate image embedding
    if image:
        inputs = clip_processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs).cpu().numpy()

    # Combine embeddings if both text and image are provided
    if text_embedding is not None and image_embedding is not None:
        combined_embedding = text_weight * text_embedding + image_weight * image_embedding
        query_embedding = combined_embedding / np.linalg.norm(combined_embedding)
    elif text_embedding is not None:
        query_embedding = text_embedding / np.linalg.norm(text_embedding)
    elif image_embedding is not None:
        query_embedding = image_embedding / np.linalg.norm(image_embedding)
    else:
        raise ValueError("At least one of text_query or image must be provided.")

    return query_embedding.astype(np.float32)

# Retrieve top-k results
def retrieve_top_k_by_type(
    query_embedding: np.ndarray, faiss_index, metadata, k: int = 5, target_type: str = "text"
) -> List[Dict]:
    query_embedding = query_embedding.astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, faiss_index.ntotal)

    # Filter results by type
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if metadata[idx]["type"] == target_type:
            results.append({"metadata": metadata[idx], "distance": dist})
            if len(results) >= k:
                break
    return results

# Format results for display
def format_results(results: List[Dict]) -> str:
    formatted = []
    for idx, result in enumerate(results, 1):
        meta = result["metadata"]
        formatted.append(f"Result {idx}: {meta.get('description', 'N/A')}")
        formatted.append(f"Similarity: {1 - result['distance']:.2f}")
    return "\n".join(formatted)

# Display image results in Streamlit
def display_results(results: List[Dict]):
    for result in results:
        meta = result["metadata"]
        st.write(f"**Description:** {meta.get('description', 'N/A')}")
        st.write(f"**Similarity:** {1 - result['distance']:.2f}")
        if "image_url" in meta:
            st.image(meta["image_url"], caption=meta.get("title", "N/A"))

# Main Streamlit app
def main():
    st.title("🔍 Multimodal Product Search Assistant")
    st.write("Search for products using text queries, images, or both!")

    # Load models, FAISS index, and metadata
    global clip_model, clip_processor
    clip_model, clip_processor = load_clip_model()
    faiss_index, metadata = load_faiss_and_metadata()

    # Input fields
    text_query = st.text_input("Enter a text query:")
    uploaded_image = st.file_uploader("Or upload an image:", type=["png", "jpg", "jpeg"])

    # Process uploaded image
    query_image = None
    if uploaded_image:
        query_image = Image.open(uploaded_image).convert("RGB")
        st.image(query_image, caption="Uploaded Image", use_column_width=True)

    # Search button
    if st.button("Search"):
        if not text_query and not query_image:
            st.error("Please provide a text query or upload an image.")
        else:
            with st.spinner("Generating query embedding..."):
                query_embedding = generate_query_embedding(text_query, query_image)

            with st.spinner("Searching FAISS index..."):
                text_results = retrieve_top_k_by_type(query_embedding, faiss_index, metadata, k=5, target_type="text")
                image_results = retrieve_top_k_by_type(query_embedding, faiss_index, metadata, k=5, target_type="image")

            st.subheader("Text Results")
            display_results(text_results)

            st.subheader("Image Results")
            display_results(image_results)

if __name__ == "__main__":
    main()
