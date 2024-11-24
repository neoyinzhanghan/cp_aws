import os
import io
import base64
from flask import Flask, jsonify, send_file, render_template_string
import h5py
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# File paths for testing
SLIDE_H5_PATH = "/media/hdd3/neo/viewer_sample_huong/390359.h5"
HEATMAP_H5_PATH = "/media/hdd3/neo/viewer_sample_huong/390359.h5" # _heatmap.h5"
TILE_SIZE = 256

# HTML Template for Viewer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HDF5 Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/openseadragon.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/openseadragon.min.css">
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 20px;
            margin: 0;
            padding: 20px;
            background-color: #2c2c2c;
            color: #fff;
        }
        #viewer-slide, #viewer-heatmap {
            width: 45%;
            height: 80vh;
            border: 2px solid #444;
        }
    </style>
</head>
<body>
    <div id="viewer-slide"></div>
    <div id="viewer-heatmap"></div>

    <script>
        const slideViewer = OpenSeadragon({
            id: "viewer-slide",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
            tileSources: {
                tileSource: {
                    type: "legacy-image-pyramid",
                    getTileUrl: (level, x, y) => `/tile/slide/${level}/${x}/${y}/`,
                    tileSize: 256,
                    width: 37670, // Replace with actual width
                    height: 22569, // Replace with actual height
                    maxLevel: 18
                }
            }
        });

        const heatmapViewer = OpenSeadragon({
            id: "viewer-heatmap",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
            tileSources: {
                tileSource: {
                    type: "legacy-image-pyramid",
                    getTileUrl: (level, x, y) => `/tile/heatmap/${level}/${x}/${y}/`,
                    tileSize: 256,
                    width: 37670, // Replace with actual width
                    height: 22569, // Replace with actual height
                    maxLevel: 18
                }
            }
        });
    </script>
</body>
</html>
"""


def retrieve_tile(h5_path, level, row, col):
    """Retrieve tile from an HDF5 file."""
    try:
        with h5py.File(h5_path, "r") as f:
            jpeg_string = base64.b64decode(f[str(level)][row, col])
            image = Image.open(io.BytesIO(jpeg_string))
            return image
    except Exception as e:
        print(f"Error retrieving tile: {e}")
        return None


@app.route("/")
def index():
    """Serve the main page with viewers for both HDF5 files."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/tile/slide/<int:level>/<int:x>/<int:y>/", methods=["GET"])
def get_slide_tile(level, x, y):
    """Get a tile from the slide HDF5 file."""
    tile = retrieve_tile(SLIDE_H5_PATH, level, x, y)
    if tile:
        img_io = io.BytesIO()
        tile.save(img_io, format="JPEG", quality=90)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")
    return "Tile not found", 404


@app.route("/tile/heatmap/<int:level>/<int:x>/<int:y>/", methods=["GET"])
def get_heatmap_tile(level, x, y):
    """Get a tile from the heatmap HDF5 file."""
    tile = retrieve_tile(HEATMAP_H5_PATH, level, x, y)
    if tile:
        img_io = io.BytesIO()
        tile.save(img_io, format="JPEG", quality=90)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")
    return "Tile not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
