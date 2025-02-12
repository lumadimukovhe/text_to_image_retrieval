import torch
import clip
import json
import os
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = None  # Load on demand
preprocess = None  # Load on demand

def load_clip():
    global clip_model, preprocess
    if clip_model is None:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load images (keep only filenames)
image_folder = "static/images/"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])

# Load existing captions or generate new ones
captions_file = "generated_captions.json"
if os.path.exists(captions_file):
    with open(captions_file, "r") as f:
        captions = json.load(f)
else:
    captions = {}

# Generate captions for images that don't have one
for img in image_files:
    if img not in captions:
        img_path = os.path.join(image_folder, img)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = blip_model.generate(**inputs)
        captions[img] = processor.decode(output[0], skip_special_tokens=True)

# Save captions
with open(captions_file, "w") as f:
    json.dump(captions, f, indent=4)

# Compute image embeddings (CLIP only)
image_embeddings = {}

def get_image_embedding(img_path):
    """Loads CLIP only when an image is being processed."""
    load_clip()  # Load CLIP if it's not already loaded
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(image).cpu().numpy()


def retrieve_images(query, top_k=5):
    load_clip()
    text_embedding = clip_model.encode_text(clip.tokenize([query]).to(device)).detach().cpu().numpy()

    # Process images one by one instead of preloading all into memory
    scores = {}
    for img in os.listdir("static/images/"):
        img_path = os.path.join("static/images/", img)
        image_embedding = get_image_embedding(img_path)  # Load only when needed
        scores[img] = np.dot(text_embedding, image_embedding.T)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Return filenames only
    filenames = [img[0] for img in sorted_images]

    return filenames
