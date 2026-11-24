# ==========================================================================================================
# Imports
# ==========================================================================================================

import glob
import json
import os
import shutil
import torch

import numpy as np


from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import CLIPModel, CLIPProcessor



# ==========================================================================================================
# Embedding Models
# ==========================================================================================================

@torch.no_grad()
def embed_images(paths, batch_size=16):
    # ---- CLIP embedding model (512-d) for image + query text ----
    device = "cpu"
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()

    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))

    return (np.vstack(vecs))



# ==========================================================================================================
# Beginning of execution
# ==========================================================================================================

def main():

    # --------------------------------------- Loader ---------------------------------------

    with open("../Data/augmented_food_recipe.json", "r") as f:
        recipes = json.load(f)

    print(f"✅ Loaded recipes:     {len(recipes)}")

    IMG_DIR  = "../Data/recipe_images"
    image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))
    print(f"✅ Images found: {len(image_paths)}")


    # --------------------------------------- Docs ---------------------------------------

    image_docs = []

    for i, (p, rec) in enumerate(zip(image_paths, recipes)):
        doc_id = f"img_{i}"

        image_docs.append(
            Document(
                # keeps retrieval results readable
                page_content=rec.get("image_description", f"description image {i}"),
                metadata={
                    "doc_id": doc_id,
                    "image_path": p,
                    "name": rec.get("name"),
                    "directions": rec.get("directions"),
                    "ingredients": rec.get("ingredients"),
                    "prep_time": rec.get("prep_time"),
                    "cook_time": rec.get("cook_time"),
                    "servings": rec.get("servings")
                },
            )
        )

    print("✅ image docs:", len(image_docs))

    # --------------------------------------- Embeddings ---------------------------------------

    embeddings = embed_images([d.metadata["image_path"] for d in image_docs]).tolist()

    # --------------------------------------- Vector Store ---------------------------------------

    DB_DIR = str("chroma_multimodal")

    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)  # Reset vector DB (important for reruns)

    image_db = Chroma(
        collection_name="food_images",
        persist_directory=DB_DIR,
    )

    image_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in image_docs],
        embeddings=embeddings,
        documents=[d.page_content for d in image_docs],
        metadatas=[d.metadata for d in image_docs],
    )

    print("✅ Image DB ready")



if __name__ == "__main__":
    main()