from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization, Flatten
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.models import Model
from keras import layers as l

def CNN(input_shape, num_classes):

	model = Sequential()
	model.add(Convolution2D(filters=16, kernel_size=(3,3), padding='same', \
							name='image_array', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(l.Convolution2D(filters=16, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(.5))

	model.add(Convolution2D(filters=16, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
	model.add(GlobalAveragePooling2D())
	#model.add(Flatten())
	model.add(Activation('softmax',name='predictions'))

	return model

def main():
	print('Hello World!')

	CNN((48, 48, 1), 7)

if __name__ == '__main__':
	main()
