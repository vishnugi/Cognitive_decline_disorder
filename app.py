from flask import Flask, render_template, request, send_from_directory
import random, os
from werkzeug.utils import secure_filename
import os
import joblib



app = Flask(__name__)  
random.seed(0)
app.config['SECRET_KEY'] = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

dir_path = os.path.dirname(os.path.realpath(__file__))

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
	return render_template('dataset.html')

@app.route('/algo', methods=['GET', 'POST'])
def algo():
	return render_template('algo.html')

@app.route('/res', methods=['GET', 'POST'])
def res():
	return render_template('r.html')


import tensorflow as tf
from tensorflow import keras
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam, Adamax

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image








from flask import Flask, render_template, send_file, request
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

@app.route('/download_pdf/<name>/<diagnosis>/<score>', methods=['GET'])
def download_pdf(name,diagnosis, score):
    # Retrieve the image based on the name or another identifier
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    image_path = os.path.join(UPLOAD_FOLDER, f"{name}.jpg")
    
    # Generate the PDF dynamically
    pdf_file = generate_pdf(image_path, name, diagnosis, score)
    
    # Send the PDF for download
    return send_file(pdf_file, as_attachment=True, download_name=f"{name}_report.pdf", mimetype='application/pdf')


def generate_pdf(image_path, name , Diagnosis, score):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Title of the PDF
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "MRI Prediction Report")

    # Add content to the PDF (name, diagnosis, tumor type, score)
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Name: {name}")
    
    c.drawString(100, 690, f"Diagnosis: {Diagnosis}")
    c.drawString(100, 670, f"Score: {score}")

    # Add the image to the PDF (adjust image path and size accordingly)
    c.drawImage(image_path, 100, 450, width=200, height=200)

    c.save()
    buffer.seek(0)  # Move to the start of the buffer
    return buffer

# def resultc():
    
#             img = cv2.imread('static/uploads/'+filename)
#             img = cv2.resize(img, (224, 224))
#             img = img.reshape(1, 224, 224, 3)
#             img = img/255.0
#             pred = covid_model.predict(img)
#             if pred < 0.5:
#                 pred = 0
#             else:
#                 pred = 1
#             # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
#             return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

#         else:
#             flash('Allowed image types are - png, jpg, jpeg')
#             return redirect(request.url)

# 1
# Load the pre-trained model
modelvgg16_loaded = load_model('models/vgg16.h5')

# Define the class labels
class_labels = ['Mild Demented', 'Moderate Dementaed', 'Non Demented', 'Very Mild Demented']  # Replace with your actual class names
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the prediction function
def pr(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Make predictions
    predictions = modelvgg16_loaded.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    probability = np.max(predictions) * 100  # Convert probability to percentage

    # Get the predicted label
    predicted_label = class_labels[predicted_class]

    return predicted_label, round(probability, 2)

# Define the route for prediction
@app.route('/pred', methods=['GET', 'POST'])
def pred():
    if request.method == 'GET':
        # Render the form to upload an image if the request is GET
        return render_template('pred.html')

    if request.method == 'POST':
        # Check if an image file is present in the request
        if 'mri_image' not in request.files:
            return render_template('pred.html', message='No image file selected')

        name = request.form.get('name')
        image = request.files['mri_image']

        if not name:
            return render_template('pred.html', message='Name is required')

        if image.filename == '':
            return render_template('pred.html', message='No image file selected')

        # Save the uploaded image
        UPLOAD_FOLDER = os.path.join('static', 'uploads')
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        image_filename = f"{name}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        image.save(image_path)

        # Predict tumor type
        predicted_label, probability = pr(image_path)

        # Render the results
        return render_template('result.html', data=[name, predicted_label, probability])


@app.route("/games")
def games():
     return  render_template("game.html")

@app.route("/voice")
def voice():
     return  render_template("voice.html")

@app.route("/game1")
def game1():
     return render_template("game1.html")

@app.route("/game2")
def game2():
     return render_template("game2.html")

@app.route("/game3")
def game3():
     return render_template("game3.html")

if __name__ == '__main__':
	app.run(debug=True)