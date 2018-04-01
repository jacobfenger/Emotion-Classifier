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
		headers = next(datareader)
		for row in datareader:
			labels.append(int(row[0]))
			pixels = list(map(int, row[1].split()))
			images.append(pixels)

	return np.asarray(images), np.asarray(labels)

def main():
	train_images, train_labels = extract_data('../../data/fer2013/train.csv')
	test_images, test_labels = extract_data('../../data/fer2013/test.csv')

	train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
	test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)

	K.set_image_dim_ordering('tf')
	train_labels = utils.to_categorical(train_labels, num_classes=7)

	model = CNN((48, 48, 1), 7)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	model.fit(train_images, train_labels, epochs=20, batch_size=128)

	#score = model.evaluate(test_images, test_labels, batch_size=128)
	#print('SCORE:', score)

if __name__ == '__main__':
	main()