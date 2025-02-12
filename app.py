from flask import Flask, render_template, request, jsonify
from models import retrieve_images
import os

app = Flask(__name__, static_folder="static")

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

    # Only return filenames, not full paths
    image_filenames = [os.path.basename(img) for img in results]

    return jsonify({"images": image_filenames})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
