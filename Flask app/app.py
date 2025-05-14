import os
import numpy as np
import cv2
from PIL import Image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from flask import Flask, render_template, request, jsonify, redirect, url_for
import io
from ultralytics import YOLO # type: ignore

app = Flask(__name__)

# Load the classification model
classification_model = load_model('model.h5')


# Load the segmentation model
segmentation_model_path = "best.pt"
segmentation_model = YOLO(segmentation_model_path)

# Define folder paths for uploads and processed images
UPLOAD_FOLDER = 'static/uploads/uploaded_image'
SEGMENTATION_FOLDER = 'static/uploads/segmentation_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTATION_FOLDER, exist_ok=True)

# Define symptom weights globally
symptom_weights = {
    "headaches": 10,
    "dizziness": 5,
    "vision": 8,
    "difficulty_walking": 7,
    "nausea": 6,
    "memory_loss": 9,
    "head_trauma": 4,
    "tinnitus": 3,
    "vision_changes": 7,
    "weight_loss": 8,
    "family_history": 10,
    "mood_changes": 5,
    "numbness": 6,
    "seizures": 9,
    "speech_problems": 8,
    "coordination": 7,
    "swelling": 4,
    "nausea_morning": 6,
    "persistent_headache": 9,
    "fever": 5
}

def img_pred(img_data):
    try:
        img = Image.open(io.BytesIO(img_data))
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(opencvImage, (150, 150))
        img_reshaped = img_resized.reshape(1, 150, 150, 3)
        p = classification_model.predict(img_reshaped)
        p = np.argmax(p, axis=1)[0]

        if p == 0:
            result = 'Glioma Tumor'
        elif p == 1:
            result = 'No Tumor'
        elif p == 2:
            result = 'Meningioma Tumor'
        else:
            result = 'Pituitary Tumor'

        return result
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    # Default values for the initial page load
    tumor_chance = None
    consultation_message = None
    return render_template('index.html', tumor_chance=tumor_chance, consultation_message=consultation_message)

@app.route('/symptom', methods=['GET', 'POST'])
def symptom():
    # Initialize the values before any form submission
    tumor_chance = None
    consultation_message = None

    if request.method == 'POST':
        # Collect all symptoms data from the form
        symptoms = {
            "headaches": request.form.get("headaches"),
            "dizziness": request.form.get("dizziness"),
            "vision": request.form.get("vision"),
            "difficulty_walking": request.form.get("difficulty_walking"),
            "nausea": request.form.get("nausea"),
            "memory_loss": request.form.get("memory_loss"),
            "head_trauma": request.form.get("head_trauma"),
            "tinnitus": request.form.get("tinnitus"),
            "vision_changes": request.form.get("vision_changes"),
            "weight_loss": request.form.get("weight_loss"),
            "family_history": request.form.get("family_history"),
            "mood_changes": request.form.get("mood_changes"),
            "numbness": request.form.get("numbness"),
            "seizures": request.form.get("seizures"),
            "speech_problems": request.form.get("speech_problems"),
            "coordination": request.form.get("coordination"),
            "swelling": request.form.get("swelling"),
            "nausea_morning": request.form.get("nausea_morning"),
            "persistent_headache": request.form.get("persistent_headache"),
            "fever": request.form.get("fever")
        }

        # Calculate the total weight of symptoms marked as "yes"
        total_weight = 0
        total_possible_weight = sum(symptom_weights.values())

        for symptom, answer in symptoms.items():
            if answer == "yes":
                total_weight += symptom_weights[symptom]

        # Calculate the tumor chance as a percentage
        if total_possible_weight > 0:  # To prevent division by zero
            tumor_chance = (total_weight / total_possible_weight) * 100
            tumor_chance = round(tumor_chance, 2)
        # Generate consultation message based on tumor chance
        if tumor_chance is not None:
            if tumor_chance >= 75:
                consultation_message = "Your symptoms suggest a high chance of a brain tumor. Please consult a doctor immediately for further testing and diagnosis."
            elif tumor_chance >= 50:
                consultation_message = "Your symptoms indicate a moderate chance of a brain tumor. It is recommended to consult a healthcare professional for further evaluation."
            elif tumor_chance >= 25:
                consultation_message = "Your symptoms suggest a low to moderate chance of a brain tumor. However, it's still advisable to seek medical advice if symptoms persist."
            else:
                consultation_message = "Your symptoms do not suggest a high risk of a brain tumor. However, if you are concerned, please consult a doctor for peace of mind."
        else:
            consultation_message = "Unable to calculate tumor chance. Please answer all questions."

    return render_template('faq/symptom.html', tumor_chance=tumor_chance, consultation_message=consultation_message)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        img_data = file.read()
        prediction = img_pred(img_data)
        return jsonify({"result": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/segment', methods=['POST'])
def segment_image():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Perform segmentation
        results = segmentation_model(file_path, save=True)

        # YOLO saves results in "runs/segment/predict". Find the output file
        output_dir = "runs/segment/predict"
        segmented_image_path = os.path.join(output_dir, os.path.basename(file_path))

        # Move the segmented image to the SEGMENTATION_FOLDER for Flask to serve
        if os.path.exists(segmented_image_path):
            final_path = os.path.join(SEGMENTATION_FOLDER, os.path.basename(file_path))
            os.makedirs(SEGMENTATION_FOLDER, exist_ok=True)
            os.replace(segmented_image_path, final_path)  # Move file
        else:
            return jsonify({"error": "Segmented image not found in the output directory"})

        # Generate a URL for the segmented image
        segmented_image_url = url_for('static', filename=f'uploads/segmentation_results/{os.path.basename(file_path)}')

        return jsonify({
            "result": "Segmentation completed successfully",
            "segmented_image_url": segmented_image_url
        })
    except Exception as e:
        return jsonify({"error": f"Can't do segmentation: {str(e)}"})


@app.route('/classification')
def classification():
    return render_template('classification/index.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation/index.html')

@app.route('/contact')
def contact():
    return render_template('Contact us/Contact.html')

@app.route('/faq')
def faq():
    return render_template('faq/index.html')

@app.route('/hospital')
def hospital():
    return render_template('hospital page/hospital_info.html')

@app.route('/details')
def details():
    return render_template('faq/details.html')

if __name__ == '__main__':
    app.run(debug=True)

