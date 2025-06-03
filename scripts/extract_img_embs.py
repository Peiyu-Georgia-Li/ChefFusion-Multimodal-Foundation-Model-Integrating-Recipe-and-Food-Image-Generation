"""Extract image embeddings for a list of image urls.

Example usage:
    python extract_img_embs.py
"""
import torch
from chefFusion import models, utils

from PIL import Image
import os
import requests
from io import BytesIO
import pickle as pkl


def extract_embeddings_for_urls(image_urls: str, emb_output_path: str, device: str = "cuda"):
    # Load model checkpoint.
    model = models.load_chefFusion("checkpoints/chefFusion_opt/", load_ret_embs=False)
    model.eval()

    visual_encoder = "openai/clip-vit-large-patch14"
    feature_extractor = utils.get_feature_extractor_for_model(
        visual_encoder, train=False
    )

    output_data = {"paths": [], "embeddings": []}
    with torch.no_grad():
        for img_url in image_urls:
            if img_url.startswith('http'):
                img = Image.open(BytesIO(requests.get(img_url).content))
            else:
                with open(img_url, 'rb') as f:
                    img = Image.open(f)

            img_tensor = utils.get_pixel_values_for_model(feature_extractor, img)
            img_tensor = img_tensor[None, ...].to(device).bfloat16()
            img_emb = model.model.get_visual_embs(img_tensor, mode="retrieval")
            img_emb = img_emb[0, 0, :].cpu()
            output_data["paths"].append(img_url)
            output_data["embeddings"].append(img_emb)
    
    with open(emb_output_path, "wb") as f:
        pkl.dump(output_data, f)


if __name__ == "__main__":
    image_urls = ['https://farm3.staticflickr.com/2045/5797966927_e82231c628_z.jpg']  # TODO: Replace with image urls, or image paths
    if image_urls == []:
        raise ValueError("Please replace `image_urls` with a list of image urls.")
    extract_embeddings_for_urls(image_urls, "recipe1m_embeddings.pkl")
