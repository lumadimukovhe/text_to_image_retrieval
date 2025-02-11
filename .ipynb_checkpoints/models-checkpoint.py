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

# Load images
image_folder = "./static/images/"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Generate captions if not already available
captions_file = "generated_captions.json"
if os.path.exists(captions_file):
    with open(captions_file, "r") as f:
        captions = json.load(f)
else:
    captions = {}

# Generate captions for new images
for img_path in image_files:
    if img_path not in captions:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = blip_model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions[img_path] = caption

# Save captions
with open(captions_file, "w") as f:
    json.dump(captions, f, indent=4)

# Encode images
image_embeddings = {}
for img_path, caption in captions.items():
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embeddings[img_path] = clip_model.encode_image(image).cpu()

def retrieve_images(query, top_k=5):
    """Retrieve top K images based on text query"""
    text_embedding = clip_model.encode_text(clip.tokenize([query]).to(device)).cpu().detach().numpy()
    scores = {img_path: np.dot(text_embedding, img_embedding.numpy().T) for img_path, img_embedding in image_embeddings.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
