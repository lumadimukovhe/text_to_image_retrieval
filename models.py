import torch
import clip
import json
import os
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load images (keep only filenames)
image_folder = "static/images/"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])

# Load captions
captions_file = "generated_captions.json"
if os.path.exists(captions_file):
    with open(captions_file, "r") as f:
        captions = json.load(f)
else:
    captions = {}

# Generate missing captions
for img in image_files:
    if img not in captions:
        image = Image.open(os.path.join(image_folder, img)).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = blip_model.generate(**inputs)
        captions[img] = processor.decode(output[0], skip_special_tokens=True)

# Save captions
with open(captions_file, "w") as f:
    json.dump(captions, f, indent=4)

# Compute image embeddings lazily
image_embeddings = {}

def get_image_embedding(img_name):
    """Load or compute image embedding lazily"""
    img_path = os.path.join(image_folder, img_name)
    if img_name in image_embeddings:
        return image_embeddings[img_name]

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image).cpu()

    image_embeddings[img_name] = embedding
    return embedding

def retrieve_images(query, top_k=5):
    """Retrieve top K images based on text query"""
    text_embedding = clip_model.encode_text(clip.tokenize([query]).to(device)).cpu().detach().numpy()

    scores = {img: np.dot(text_embedding, get_image_embedding(img).numpy().T) for img in image_files}
    sorted_images = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Return filenames only
    filenames = [img[0] for img in sorted_images]

    print("\nDebugging `retrieve_images()` - Sending Filenames:", filenames)

    return filenames
