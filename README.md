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
✅ **Deployable on Render & Heroku.**  

---
## Project Structure
text_to_image_retrieval/
│── app.py                # Flask Backend
│── models.py             # Model Logic (CLIP + BLIP)
│── static/
│   ├── images/           # Image Storage
│   ├── default.jpg       # Fallback Image
│── templates/
│   ├── index.html        # Bootstrap Frontend
│── generated_captions.json  # Pre-generated Image Captions
│── requirements.txt      # Dependencies
│── Procfile              # Required for Heroku Deployment
│── render.yaml           # Required for Render Deployment
│── architecture.png      # System Architecture Diagram
│── README.md             # Project Documentation
---
## 🛠️ Installation & Setup

### **1 Clone the Repository**
```bash
git clone https://github.com/lumadimukovhe/text_to_image_retrieval.git
cd text_to_image_retrieval

### **2 Create Virtual Environment**
python -m venv venv

### **3 Activate the Virtual Environment**
source venv\Scripts\activate

### **4 Istall dependencies**
pip install -r requirements.txt

### **5 Run Flask APP**
python app.py

The app will be available at: http://127.0.0.1:5000/
---
### Assumptions & Challanges
#### Assumptions:
All images are stored in JPEG (.jpg) format.
The dataset contains high-quality images.
CLIP embeddings accurately represent image features.
Challenges Faced:
1️. Image Path Issues:

Initially, odd-indexed images failed to load due to incorrect path formatting (static/static/images/).
Fixed by ensuring only filenames are returned from the backend.

2. Performance Optimization:

Lazy loading of embeddings improved the speed of retrieval.
---
