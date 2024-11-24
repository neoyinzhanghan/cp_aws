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


def image_to_jpeg_string(image):
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()
    try:
        # Save the image in JPEG format to the buffer
        image.save(buffer, format="JPEG")
        jpeg_string = buffer.getvalue()  # Get the byte data
    finally:
        buffer.close()  # Explicitly close the buffer to free memory

    return jpeg_string


def jpeg_string_to_image(jpeg_string):
    # Create an in-memory bytes buffer from the byte string
    buffer = io.BytesIO(jpeg_string)

    # Open the image from the buffer and keep the buffer open
    image = Image.open(buffer)

    # Load the image data into memory so that it doesn't depend on the buffer anymore
    image.load()

    return image


def encode_image_to_base64(jpeg_string):
    return base64.b64encode(jpeg_string)


def decode_image_from_base64(encoded_string):
    return base64.b64decode(encoded_string)


def retrieve_tile_h5(h5_path, level, row, col):
    with h5py.File(h5_path, "r") as f:
        try:
            jpeg_string = f[str(level)][row, col]
            jpeg_string = decode_image_from_base64(jpeg_string)
            image = jpeg_string_to_image(jpeg_string)

        except Exception as e:
            print(
                f"Error: {e} occurred while retrieving tile at level: {level}, row: {row}, col: {col} from {h5_path}"
            )
            jpeg_string = f[str(level)][row, col]
            print(f"jpeg_string: {jpeg_string}")
            jpeg_string = decode_image_from_base64(jpeg_string)
            print(f"jpeg_string base 64 decoded: {jpeg_string}")
            raise e
        return image


@app.route("/")
def index():
    """Serve the main page with viewers for both HDF5 files."""
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
