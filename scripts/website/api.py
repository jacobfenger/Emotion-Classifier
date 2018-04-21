import flask
import sys
from flask import Flask, render_template, request
import keras
from keras import models
from keras.preprocessing.image import img_to_array
import preprocess
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
@app.route("/index")
def index():
	# Return an html file called index.html located in templates
	return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=="POST":

		file = request.files['image']
		filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(filename)

		if not file:
			return render_template('index.html', label="No file")

		################## PREPROCESSING #############################
		image = cv2.imread('./' + filename)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		casc_path = 'haarcascade_frontalface_default.xml'
		faceCascade = cv2.CascadeClassifier(casc_path)

		# Detect a face in the image
		faces = faceCascade.detectMultiScale(gray, \
											scaleFactor=1.1, \
											minNeighbors=5, \
   											minSize=(30, 30), \
    										flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

		i = 0
		face_array = []
		# Crop the face from the image
		for (x, y, w, h) in faces:
			i += 1
			#cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			crop_img = image[y:y+h, x:x+w]
			crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
			face_array.append(np.asarray(crop_img))


		array = img_to_array(face_array[0])  # this is a Numpy array with shape (3, 150, 150)
		arrayresized = cv2.resize(array, (48, 48))
		inputarray = arrayresized[np.newaxis,..., np.newaxis]

		#################### PREDICTION ###################################

		# YOU MUST USE THE THEANO BACKEND FOR THIS TO WORK
		# TENSORFLOW BREAKS IT WHEN TRYING TO LOAD A MODEL
		model = keras.models.load_model('intro_model.h5') # Load the model
		label = model.predict(inputarray) # Predict
		label = np.argmax(label[0]) # Take the argmax (Most likely label)

		if label == 0:
			label = 'Angry'
		elif label == 1:
			label = 'Disgust'
		elif label == 2:
			label = 'Fear'
		elif label == 3:
			lael = 'Happy'
		elif label == 4:
			label = 'Sad'
		elif label == 5:
			label = 'Surprise'
		else: 
			label = 'Netural'

		# Label must be a string I guess
		return render_template('index.html', label=str(label), file=file)

if __name__ == '__main__':
	model = models.load_model('intro_model.h5')
	app.run(host='0.0.0.0', port=8000, debug=True)

