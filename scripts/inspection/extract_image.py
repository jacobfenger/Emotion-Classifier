# Script that will convert the pixel data in the data set to a viewable image
# 
# Thanks to use 'iftekharanam' in the Kaggle discussions for saving me a few mins:
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/discussion/29428
#

import os
import csv
import numpy as np 
import scipy.misc 

with open('../data/fer2013/train.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	headers = datareader.next()
	row = datareader.next()

	emotion = row[0]
	pixels = map(int, row[1].split())

	pixels_array = np.asarray(pixels)

	image = pixels_array.reshape(48, 48)
	# This is only necessary to get rbg values, 
	# but since it's already greyscale it doesn't matter.
	stacked_image = np.dstack((image,) * 3) 
	scipy.misc.imsave('./face.jpg', stacked_image)