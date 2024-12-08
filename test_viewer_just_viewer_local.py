import os
import io
import base64
from flask import Flask, jsonify, send_file, render_template_string
import h5py
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# File path for testing
SLIDE_H5_PATH = "/media/ssd2/neo/cp_aws_playground/test_neo.h5"
TILE_SIZE = 256

# Get the level 0 width and height of the slide
with h5py.File(SLIDE_H5_PATH, "r") as f:
    level_0_width = f["level_0_width"][0]
    level_0_height = f["level_0_height"][0]

# HTML Template for Viewer with placeholders for dynamic values
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
            margin: 0;
            padding: 20px;
            background-color: #2c2c2c;
            color: #fff;
        }
        #viewer-slide {
            width: 80%;
            height: 80vh;
            border: 2px solid #444;
        }
    </style>
</head>
<body>
    <div id="viewer-slide"></div>

    <script>
        OpenSeadragon({
            id: "viewer-slide",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
            tileSources: {
                type: "legacy-image-pyramid",
                getTileUrl: (level, x, y) => `/tile/slide/${level}/${x}/${y}/`,
                tileSize: 256,
                width: {{ width }},
                height: {{ height }},
                maxLevel: 18
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


@app.route("/")
def index():
    """Serve the main page with the slide viewer."""
    # Render the template with dynamic width and height
    rendered_template = HTML_TEMPLATE.replace("{{ width }}", str(level_0_width)).replace("{{ height }}", str(level_0_height))
    return render_template_string(rendered_template)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
