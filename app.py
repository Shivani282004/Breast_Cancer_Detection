from flask import Flask, render_template, request,redirect,url_for,send_from_directory
import joblib
import numpy as np
from io import BytesIO
import pdfplumber
import pytesseract
import time
from werkzeug.utils import secure_filename
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import re

app = Flask(__name__)
app.url_map.strict_slashes = False

# Load the trained model
model = load_model("BREAST_cancer_PREDICTION_model.h5")

# Define image size for the model
image_size = (128, 128)

# Define classes
classes = ["benign", "malignant", "normal"]
# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the 'uploads' folder exists, if not, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

cancer_model = joblib.load("cancer_model.joblib")  


feature_columns = [
    "id",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]


def extract_text(file_content):
    text = ""
    with pdfplumber.open(BytesIO(file_content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                image = page.to_image()
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text + "\n"
    return text


def extract_features(text):
    extracted_features = {"id": 0}  
    for feature in feature_columns[1:]:
        match = re.search(fr"-?\s*{feature}\s*[:=]?\s*([\d.]+)", text, re.IGNORECASE)
        if match:
            extracted_features[feature] = float(match.group(1))
        else:
            extracted_features[feature] = None 
            app.logger.debug(f"Missing feature: {feature}")
    if len(extracted_features) != len(feature_columns):
        raise ValueError("Extracted features count does not match expected features count.")
    return extracted_features


def predict_cancer(features):
    feature_values = [features["id"]] + [features[col] for col in feature_columns[1:]]
    feature_values = np.array(feature_values).reshape(1, -1)
    prediction = cancer_model.predict(feature_values)
    return "Malignant" if prediction[0] == 1 else "Benign"


@app.route('/predict_res', methods=['POST'])
def predict_res():
    try:
        features = [float(request.form[f'value{i}']) for i in range(1, 31)]
        data_points = [0] + features
        data_np = np.array(data_points, dtype=float).reshape(1, -1)
        start_time = time.time()
        prediction = cancer_model.predict(data_np)
        prediction_time = round(time.time() - start_time, 9)

        
        probabilities = getattr(cancer_model, "predict_proba", lambda x: None)(data_np)
        accuracy = max(probabilities[0]) if probabilities is not None and len(probabilities[0]) > 0 else 1.0

        
        output = "Malignant" if prediction[0] == 1 else "Benign"
        return render_template('result.html', output=output, accuracy=accuracy, time=prediction_time)
    except Exception as e:
        app.logger.error(f"Error in /predict_res: {e}")
        return f"An error occurred: {str(e)}", 500


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'report' not in request.files or request.files['report'].filename == '':
            return "No file selected", 400

        file_content = request.files['report'].read()
        extracted_text = extract_text(file_content)
        feature_data = extract_features(extracted_text)

        if all(feature_data[col] is not None for col in feature_columns):
            start_time = time.time()

           
            cancer_status = predict_cancer(feature_data)
            feature_values = [feature_data[col] for col in feature_columns]
            feature_array = np.array(feature_values).reshape(1, -1)

            try:
                probabilities = cancer_model.predict_proba(feature_array)[0]
                accuracy = max(probabilities)
            except AttributeError:
                accuracy = 1.0  

            prediction_time = round(time.time() - start_time, 9)

            return render_template(
                'result.html',
                output=cancer_status,
                accuracy=accuracy,
                time=prediction_time
            )
        else:
            return render_template(
                'result.html',
                output="Incomplete data extracted. Ensure all fields are present.",
                accuracy=0,
                time=0
            )
    except Exception as e:
        app.logger.error(f"Error in /upload: {e}")
        return str(e), 500




@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part', 400
    
    file = request.files['image']
    
    if file.filename == '':
        return 'No selected file', 400
    
    # Process the image
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Start timer to measure prediction time
        start_time = time.time()
        
        # Load and preprocess the image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, image_size)
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]
        accuracy = np.max(prediction)  # Get the highest probability for accuracy
        
        # Calculate the time taken for prediction
        end_time = time.time()
        time_taken = end_time - start_time
        
        return render_template('result2.html', prediction=predicted_class, accuracy=accuracy, time=time_taken)
    
    return 'Invalid file format', 400

# Check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)