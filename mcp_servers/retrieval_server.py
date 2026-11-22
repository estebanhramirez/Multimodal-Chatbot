# ==========================================================================================================
# Imports
# ==========================================================================================================

import base64
import io
import torch
import re

import numpy as np

from fastmcp import FastMCP
from langchain_chroma import Chroma
from PIL import Image
from transformers import CLIPModel, CLIPProcessor



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

@torch.no_grad()
def embed_images(images, batch_size=16):
    """images: list of file paths (str / Path) and/or PIL.Image objects."""
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
    """
        Parse the output of Chromadb.retrieve:
        Chroma returns lists-of-lists; unwrap the first query.

        Args:
            res (dict): raw result from invoking Chromadb.retrieve()
        Returns:
            Tuple[List]: ids, docs, metas, dists
    """
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return (ids, docs, metas, dists)


def _decode_image(query_image: str) -> Image.Image:
    """Decode a base64 string (optionally a data URL) into an RGB PIL image."""
    # Strip a possible "data:image/png;base64," prefix
    query_image = re.sub(r"^data:image/[^;]+;base64,", "", query_image)
    try:
        img_bytes = base64.b64decode(query_image, validate=True)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"query_image is not a valid base64-encoded image: {e}")



# ==========================================================================================================
# MCP Server
# ==========================================================================================================

# --------------------------------------------- Server declaration -----------------------------------------------------

server = FastMCP(name="Retrieval_server", instructions="Server for retrieval of restaurants and recipes.")

# --------------------------------------------- MCP tools -----------------------------------------------------

@server.tool()
def retrieve_images_by_image(chromadb_dir: str, query_image: str, k: int = 5, where: dict | None = None):
    """
    Retrieve similar images from the image database based on a query image,
    with optional metadata filtering.

    Args:
        chromadb_dir (str): the path to the database.
        query_image (str): the query image, encoded as a base64 string.
        k (int): the number of top results to return.
        where (dict | None): Optional metadata filter, e.g. {"cuisine": "Italian"}.
    Returns:
        Tuple[List]: ids, docs, metas, dists
    """
    chromadb = Chroma(
        collection_name="food_images",
        persist_directory=chromadb_dir,
    )

    img = _decode_image(query_image)
    q_vec = embed_images([img])[0]  # 512-d, cosine-ready

    res = chromadb._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
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