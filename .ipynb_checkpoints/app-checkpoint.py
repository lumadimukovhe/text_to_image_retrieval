from flask import Flask, render_template, request, jsonify
from models import retrieve_images

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "Please enter a search query."})

    results = retrieve_images(query, top_k)
    image_paths = [img[0].replace("./static/", "") for img in results]  # Format paths for frontend

    return jsonify({"images": image_paths})

if __name__ == "__main__":
    app.run(debug=True)
