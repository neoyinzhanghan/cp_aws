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
HEATMAP_H5_PATH = "/media/hdd3/neo/viewer_sample_huong/390359_heatmap.h5"
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
        #viewer-slide, #viewer-heatmap, #viewer-overlay {
            width: 30%;
            height: 80vh;
            border: 2px solid #444;
        }
    </style>
</head>
<body>
    <div id="viewer-slide"></div>
    <div id="viewer-heatmap"></div>
    <div id="viewer-overlay"></div>

    <script>
        OpenSeadragon({
            id: "viewer-slide",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
            tileSources: {
                tileSource: {
                    type: "legacy-image-pyramid",
                    getTileUrl: (level, x, y) => `/tile/slide/${level}/${x}/${y}/`,
                    tileSize: 256,
                    width: 37670, // Update with actual width
                    height: 22569, // Update with actual height
                    maxLevel: 18
                }
            }
        });

        OpenSeadragon({
            id: "viewer-heatmap",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
            tileSources: {
                tileSource: {
                    type: "legacy-image-pyramid",
                    getTileUrl: (level, x, y) => `/tile/heatmap/${level}/${x}/${y}/`,
                    tileSize: 256,
                    width: 37670, // Update with actual width
                    height: 22569, // Update with actual height
                    maxLevel: 18
                }
            }
        });

        OpenSeadragon({
            id: "viewer-overlay",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
            tileSources: {
                tileSource: {
                    type: "legacy-image-pyramid",
                    getTileUrl: (level, x, y) => `/tile/overlay/${level}/${x}/${y}/`,
                    tileSize: 256,
                    width: 37670, // Update with actual width
                    height: 22569, // Update with actual height
                    maxLevel: 18
                }
            }
        });
    </script>
</body>
</html>
"""


def retrieve_tile_h5(h5_path, level, row, col):
    with h5py.File(h5_path, "r") as f:
        try:
            jpeg_string = f[str(level)][row, col]
            image = Image.open(io.BytesIO(base64.b64decode(jpeg_string)))
            image.load()
            return image
        except Exception as e:
            print(f"Error retrieving tile: {e}")
            return None


def overlay_tiles(slide_tile, heatmap_tile, alpha=0.5):
    """Overlay heatmap tile on slide tile with specified alpha."""
    if slide_tile is None or heatmap_tile is None:
        return None
    slide_tile = slide_tile.convert("RGBA")
    heatmap_tile = heatmap_tile.convert("RGBA")
    blended = Image.blend(slide_tile, heatmap_tile, alpha)
    # Convert to RGB to remove alpha channel
    blended = blended.convert("RGB")
    return blended


@app.route("/")
def index():
    """Serve the main page with viewers for slide, heatmap, and overlay."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/tile/slide/<int:level>/<int:x>/<int:y>/", methods=["GET"])
def get_slide_tile(level, x, y):
    """Get a tile from the slide HDF5 file."""
    tile = retrieve_tile_h5(SLIDE_H5_PATH, level, x, y)
    if tile:
        img_io = io.BytesIO()
        tile.save(img_io, format="JPEG", quality=90)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")
    return "Tile not found", 404


@app.route("/tile/heatmap/<int:level>/<int:x>/<int:y>/", methods=["GET"])
def get_heatmap_tile(level, x, y):
    """Get a tile from the heatmap HDF5 file."""
    tile = retrieve_tile_h5(HEATMAP_H5_PATH, level, x, y)
    if tile:
        img_io = io.BytesIO()
        tile.save(img_io, format="JPEG", quality=90)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")
    return "Tile not found", 404


@app.route("/tile/overlay/<int:level>/<int:x>/<int:y>/", methods=["GET"])
def get_overlay_tile(level, x, y):
    """Get a tile that overlays heatmap on slide."""
    slide_tile = retrieve_tile_h5(SLIDE_H5_PATH, level, x, y)
    heatmap_tile = retrieve_tile_h5(HEATMAP_H5_PATH, level, x, y)
    overlay_tile = overlay_tiles(slide_tile, heatmap_tile, alpha=0.5)
    if overlay_tile:
        img_io = io.BytesIO()
        overlay_tile.save(img_io, format="JPEG", quality=90)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")
    return "Tile not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
