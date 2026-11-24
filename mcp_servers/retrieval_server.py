# ==========================================================================================================
# Imports
# ==========================================================================================================

import torch

import numpy as np

from fastmcp import FastMCP
from langchain_chroma import Chroma
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

def embed_texts(texts, batch_size=64):
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)


@torch.no_grad()
def embed_images(images, batch_size=16):
    device = "cpu"
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()

    def _load(x):
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return Image.open(x).convert("RGB")

    vecs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        imgs = [_load(x) for x in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))

    return np.vstack(vecs)


def _unwrap(res: dict):
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return (ids, docs, metas, dists)



# ==========================================================================================================
# MCP Server
# ==========================================================================================================

# --------------------------------------------- Server declaration -----------------------------------------------------

server = FastMCP(name="Retrieval_server", instructions="Server for retrieval of restaurants and recipes.")

# --------------------------------------------- MCP tools -----------------------------------------------------

@server.tool()
def retrieve_user_reviews(query: str):
    """
    Search customer reviews of restaurants by meaning and return the most relevant ones.

    Use this tool when the user wants first-hand opinions or experiences: what
    diners thought of a place or dish, service quality, atmosphere, prices, or
    complaints and praise. For editorial articles, guides, or general
    recommendations, use retrieve_articles instead.

    Args:
        query: Natural-language search text describing what to look for,
            e.g. "slow service and long waits" or "is the carbonara good".

    Returns:
        A dict with parallel lists: "ids", "documents", "metadatas", "distances". Lower distance means more similar.
    """

    chromadb = Chroma(
        collection_name="user_reviews",
        persist_directory="../chroma_multimodal",
    )

    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=2,
        where=None,
        include=["documents", "metadatas", "distances"],
    )

    return (_unwrap(res))


@server.tool()
def find_restaurants(query: str):
    """
    Search restaurant articles by meaning and return the most relevant ones.

    Use this tool whenever the user asks about restaurants, cuisines, dishes,
    reviews, or dining recommendations that articles in the database might cover.

    Args:
        query: Natural-language search text, e.g. "best ramen in Tokyo".

    Returns:
        A dict with parallel lists: "ids", "documents", "metadatas", "distances". Lower distance means more similar.
    """

    chromadb = Chroma(
        collection_name="restaurant_articles",
        persist_directory="../chroma_multimodal",
    )

    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=5,
        where=None,
        include=["documents", "metadatas", "distances"],
    )

    return (_unwrap(res))


@server.tool()
def find_similar_food_images(query_image: str):
    """
    Find food images in the database that are visually similar to a given image,
    and return their stored descriptions and metadata (e.g. dish name, source).

    Use this tool whenever the user provides a food image (or an image file path)
    and wants to identify the dish, find ingredients, or look up other details about it.

    Args:
        query_image: Absolute local file path to the query image (e.g. "/home/user/images/pizza.jpg"). Not base64 and not a description.

    Returns:
        A dict with parallel lists: "ids", "documents", "metadatas", "distances". Lower distance means more similar.
    """

    chromadb = Chroma(
        collection_name="food_images",
        persist_directory="../chroma_multimodal",
    )

    q_vec = embed_images([query_image])[0]  # 512-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=1,
        where=None,
        include=["documents", "metadatas", "distances"],
    )
    return (_unwrap(res))



# ==========================================================================================================
# Entry point
# ==========================================================================================================

# Run the Server with HTTP transport
if __name__ == "__main__":
    print("Starting HTTP MCP Server on http://127.0.0.1:8080")

    server.run(transport="http", host="127.0.0.1", port=8080)