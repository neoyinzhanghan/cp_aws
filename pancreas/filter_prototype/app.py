from flask import Flask, request, jsonify
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the metadata
metadata_path = "pancreas_metadata.csv"  # Ensure this file exists in the same folder
metadata = pd.read_csv(metadata_path)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pancreas Metadata</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <h1>Pancreas Metadata Viewer</h1>

    <label for="group">Select Group:</label>
    <select id="group" name="group">
        <option value="" disabled selected>Select a group</option>
    </select>

    <div id="slide-menu" style="display: none;">
        <label for="slide">Select Slide:</label>
        <select id="slide" name="slide">
            <option value="" disabled selected>Select a slide</option>
        </select>
    </div>

    <div id="result"></div>

    <script>
        $(document).ready(function() {
            // Populate group dropdown
            $.ajax({
                url: '/get_groups',
                method: 'GET',
                success: function(data) {
                    const groupDropdown = $('#group');
                    data.forEach(group => {
                        groupDropdown.append(`<option value="${group}">${group}</option>`);
                    });
                }
            });

            $('#group').change(function() {
                const selectedGroup = $(this).val();
                $('#slide-menu').hide();
                $('#result').empty();

                $.ajax({
                    url: '/get_slides',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ group: selectedGroup }),
                    success: function(data) {
                        const slideDropdown = $('#slide');
                        slideDropdown.empty().append('<option value="" disabled selected>Select a slide</option>');
                        data.forEach(slide => {
                            slideDropdown.append(`<option value="${slide.display_name}">${slide.display_name}</option>`);
                        });
                        $('#slide-menu').show();
                    }
                });
            });

            $('#slide').change(function() {
                const selectedSlide = $(this).val();
                $('#result').empty();

                $.ajax({
                    url: '/select_slide',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ display_name: selectedSlide }),
                    success: function(data) {
                        const resultDiv = $('#result');
                        resultDiv.html('<h2>Selected Slide Information:</h2><pre>' + JSON.stringify(data, null, 2) + '</pre>');
                    }
                });
            });
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML_TEMPLATE


@app.route("/get_groups", methods=["GET"])
def get_groups():
    # Ensure groups are sorted by group_order
    sorted_metadata = metadata.sort_values(by="group_order")
    groups = sorted_metadata["group"].dropna().unique().tolist()
    return jsonify(groups)


@app.route("/get_slides", methods=["POST"])
def get_slides():
    selected_group = request.json.get("group")
    filtered_metadata = metadata[metadata["group"] == selected_group]
    slides = (
        filtered_metadata[["display_name"]].drop_duplicates().to_dict(orient="records")
    )
    return jsonify(slides)


@app.route("/select_slide", methods=["POST"])
def select_slide():
    selected_display_name = request.json.get("display_name")
    # Filter the row based on the selected display_name
    row = metadata[metadata["display_name"] == selected_display_name]

    if not row.empty:
        # Include only the required columns
        selected_data = (
            row[
                [
                    "benign_prob",
                    "case_name",
                    "malignant_prob",
                    "non_diagnosis_prob",
                    "pred",
                ]
            ]
            .iloc[0]
            .to_dict()
        )
        return jsonify(selected_data)
    else:
        return jsonify({"error": "Slide not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
