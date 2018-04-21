# Script to deal with the preprocessing of an image when users upload a photo
# Some of this was taken from the following tutorial:
# https://realpython.com/face-recognition-with-python/
import cv2
import numpy as np

# Perform face detection on images and return all faces found
def preprocess_image(image):
	casc_path = 'haarcascade_frontalface_default.xml'

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(casc_path)

	image = cv2.imread(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

		# Display image to screen
		# cv2.imshow('', crop_img)
		# cv2.waitKey(0)

		# Append found face to a list
		face_array.append(crop_img)

	return face_array

def main():
	# Path to images for testing purposes
	img_path = 'ryan.jpg'

	faces = preprocess_image(img_path)

	# Once the faces are found. We must then pass each one through our trained
	# CNN to determine the emotion. This is done by saving our model as a pickle.


if __name__ == '__main__':
	main()