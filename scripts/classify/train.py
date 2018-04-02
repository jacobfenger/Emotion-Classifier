from cnn import CNN
import csv 
import numpy as np
from keras import backend as K
from keras import utils 

def extract_data(filepath):
	images = []
	labels = []

	with open(filepath, 'r') as csvfile:
		datareader = csv.reader(csvfile, delimiter=',')

		# The header is the first line of the CSV, so we skip it
		headers = next(datareader) 
		for row in datareader:
			# Append the emotional label
			labels.append(int(row[0]))

			# Make the pixels an actual list of integers
			pixels = list(map(int, row[1].split())) 
			images.append(pixels)

	# Return the data as Numpy arrays
	return np.asarray(images), np.asarray(labels)

def main():
	# Get train/test data
	train_images, train_labels = extract_data('../../data/fer2013/train.csv')
	test_images, test_labels = extract_data('../../data/fer2013/test.csv')

	# Reshape the images so they are (48, 48, 1) (1 since it is grayscale)
	train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
	test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)

	# Tensorflow order (Row, Col, Channel)
	K.set_image_dim_ordering('tf')

	# Make the labels categorical arrays. Required for how the model is setup
	train_labels = utils.to_categorical(train_labels, num_classes=7)
	test_labels = utils.to_categorical(test_labels, num_classes=7)

	# Create a new model with (48, 48, 1) input shape and 7 classes
	model = CNN((48, 48, 1), 7)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary() # Output a summary of the model
	model.fit(train_images, train_labels, epochs=1, batch_size=128)

	# Evaluate the model with the test set of images
	# Change this to predict() to predict actual class labels
	score = model.evaluate(test_images, test_labels, batch_size=128)
	print('SCORE:', score)

if __name__ == '__main__':
	main()
