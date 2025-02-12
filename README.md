# Multimodal Image Retrieval (Text-to-Image Search)

## Overview
This project implements **Multimodal Image Retrieval** using **CLIP** and **BLIP** models.  
It allows users to **input a text query** (e.g., "a sunset over mountains") and retrieves the **most relevant images**.

---

## Features
✅ **Text-to-Image Search:** Retrieves relevant images based on a text query.  
✅ **CLIP & BLIP Models:** Uses CLIP for text-image matching and BLIP for image captioning.  
✅ **Bootstrap UI:** Modern and responsive frontend.  
✅ **Flask API:** Lightweight backend server.  


---
## Project Structure
#### text_to_image_retrieval/
##### 1.  Flask Backend
app.py               
##### 2. Model Logic (CLIP + BLIP)
models.py             
##### 3. Image Storage
static/
    images/           
    default.jpg      
##### 4. Bootstrap Frontend
templates/
    index.html        
##### 5. Pre-generated Image Captions
generated_captions.json  
##### 6. Dependencies
requirements.txt      
##### 7. README.md           
---
## 🛠️ Installation & Setup

##### 1. Clone the Repository

- git clone https://github.com/lumadimukovhe/text_to_image_retrieval.git
- cd text_to_image_retrieval

##### 2. Create Virtual Environment
- python -m venv venv

##### 3. Activate the Virtual Environment
- source venv\Scripts\activate

##### 4. Istall dependencies
- pip install -r requirements.txt

##### 5. Run Flask APP
- python app.py

- The app will be available at: http://127.0.0.1:5000/
---
### Assumptions & Challanges
#### Assumptions:
- All images are stored in JPEG (.jpg) format.
- The dataset contains high-quality images.
- CLIP embeddings accurately represent image features.
#### Challenges Faced:
1️. Image Path Issues:

- Initially, odd-indexed images failed to load due to incorrect path formatting (static/static/images/).
- Fixed by ensuring only filenames are returned from the backend.

2. Performance Optimization:

- Lazy loading of embeddings improved the speed of retrieval.
---

