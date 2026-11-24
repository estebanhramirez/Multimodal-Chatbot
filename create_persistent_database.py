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
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor



DB_DIR = str("chroma_multimodal")

# ==========================================================================================================
# Embedding Models
# ==========================================================================================================

def embed_texts(texts, batch_size=64):
    # ---- Text embedding model (384-d) ----
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)


@torch.no_grad()
def embed_images(paths, batch_size=16):
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

def index_reviews_by_content():

    # --------------------------------------- Loader ---------------------------------------

    with open("../Data/augmented_user_review.json", "r") as f:
        user_reviews = json.load(f)

    print(f"✅ Loaded reviews: {len(user_reviews)}")

    # --------------------------------------- Docs ---------------------------------------

    reviews_docs = []

    for i, r in enumerate(user_reviews):
        text = (
            f"Title: {r.get('title','')}\n"
            f"Text: {r.get('text','')}\n"
            f"Rating: {r.get('rating','')}"
        )

        doc_id = f"rest_{i}"

        reviews_docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "reviewId": r.get("reviewId"),
                    "userId": r.get("userId"),
                    "date": r.get("date"),
                    "images": r.get("images") or ["none"],
                    "image_captions": r.get("image_captions") or ["none"],
                    "language": r.get("language"),
                },
            )
        )

    print("✅ reviews docs:", len(reviews_docs))

    # --------------------------------------- Embeddings ---------------------------------------

    embeddings = embed_texts([d.page_content for d in reviews_docs]).tolist()

    # --------------------------------------- Vector Store ---------------------------------------

    reviews_db = Chroma(collection_name="user_reviews", persist_directory=DB_DIR)

    reviews_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in reviews_docs],
        embeddings=embeddings,
        documents=[d.page_content for d in reviews_docs],
        metadatas=[d.metadata for d in reviews_docs],
    )

    print("✅ Reviews DB ready")


def index_restaurants_by_content():

    # --------------------------------------- Loader ---------------------------------------

    with open("../Data/structured_restaurant_data.json", "r") as f:
        restaurants = json.load(f)

    print(f"✅ Loaded restaurants: {len(restaurants)}")

    # --------------------------------------- Docs ---------------------------------------

    article_docs = []

    for i, r in enumerate(restaurants):
        name = str(r.get("name", "")).strip()
        if not name:
            continue

        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style','')}\n"
            f"Location: {r.get('location','')}"
        )

        doc_id = f"rest_{i}"

        article_docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "cuisine": r.get("food_style"),
                    "location": r.get("location"),
                    "source": "restaurant",
                },
            )
        )

    print("✅ article docs:", len(article_docs))

    # --------------------------------------- Embeddings ---------------------------------------

    embeddings = embed_texts([d.page_content for d in article_docs]).tolist()

    # --------------------------------------- Vector Store ---------------------------------------

    article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)

    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=embeddings,
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )

    print("✅ Article DB ready")


def index_recipes_by_images():

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

    image_db = Chroma(collection_name="food_images", persist_directory=DB_DIR)

    image_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in image_docs],
        embeddings=embeddings,
        documents=[d.page_content for d in image_docs],
        metadatas=[d.metadata for d in image_docs],
    )

    print("✅ Image DB ready")



def main():
    index_recipes_by_images()
    index_restaurants_by_content()
    index_reviews_by_content()



if __name__ == "__main__":
    main()