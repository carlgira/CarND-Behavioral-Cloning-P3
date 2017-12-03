import pandas as pd
import numpy as np
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Lambda, Dropout, Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.regularizers import l2

img_rows = 32
img_cols = 64
image_channels = 1

max_steering_angle = 25.0

def transform_image(img):
	image = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[:, :, 1]
	image = image.reshape(160, 320, 1)
	image = cv2.resize(image, (img_cols, img_rows))[12:img_rows-4,]
	return np.reshape(image, (16, img_cols, 1))



def prepare_data():
	#files = ['run1/driving_log.csv', 'run3/driving_log.csv']
	files = ['my_data/driving_log.csv']

	driving_log = pd.read_csv(files[0], names=['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed' ])

	for i in range(1, len(files)):
		driving_log = driving_log.append(pd.read_csv(files[i], names=['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed' ]))

	num_bins = 30
	steering_angle = driving_log.steering_angle
	steering_angle_hist, steering_angle_bins = np.histogram(steering_angle, num_bins)

	count_bins = []
	for i in range(len(steering_angle_bins)-1):
		count_bins.append(len(driving_log.steering_angle[(driving_log.steering_angle > steering_angle_bins[i]) & (driving_log.steering_angle <= steering_angle_bins[i+1])  ]))

	cut_line = int(np.mean(count_bins)/5)

	driving_log_clean = driving_log

	for i in range(len(count_bins)):
		bin_samples = driving_log_clean[(driving_log_clean.steering_angle > steering_angle_bins[i]) & (driving_log_clean.steering_angle <= steering_angle_bins[i+1])  ]
		if len(bin_samples) > cut_line:
			driving_log_clean = driving_log_clean.drop(bin_samples.sample(len(bin_samples) - cut_line).index)

		x_data = []
	y_data = []

	for index, row in driving_log_clean.iterrows():
		# Center Image
		x_data.append(row.center_img)
		y_data.append(row.steering_angle)
		#y_data.append([row.speed/max_speed, (row.steering_angle)/max_steering_angle])

		# Left Image
		x_data.append(row.left_img)
		#y_data.append([row.speed/max_speed, (row.steering_angle + 0.25)/max_steering_angle])
		y_data.append((row.steering_angle + 0.2))

		# Right Image
		x_data.append(row.right_img)
		#y_data.append([row.speed/max_speed, (row.steering_angle - 0.25)/max_steering_angle])
		y_data.append((row.steering_angle - 0.2))



	x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.20)

	return x_train, x_val, y_train, y_val


def data_generator(x_data, y_data, batch_size):
	num_samples = len(x_data)
	while 1:
		for offset in range(0, num_samples, batch_size):
			batch_samples_x = x_data[offset:offset+batch_size]

			images = []
			for file_name in batch_samples_x:
				image = transform_image(cv2.imread(file_name))
				images.append(image)

			x_data_batch = np.array(images)
			y_data_batch = np.array(y_data[offset:offset+batch_size])

			yield shuffle(x_data_batch, y_data_batch)


'''
	model = Sequential()
	model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(16, img_cols, image_channels)))
	model.add(Conv2D(2, 3, 3, border_mode='valid', activation='elu'))
	model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
	model.add(Dropout(0.75))
	model.add(Flatten())
	model.add(Dense(1))
'''
def nn_model(input_shape):
	model = Sequential()

	# Normalize
	model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=input_shape))

	# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
	model.add(Conv2D(24, 5, 5, W_regularizer=l2(0.001), activation='elu'))
	model.add(Conv2D(36, 5, 5, W_regularizer=l2(0.001), activation='elu'))
	model.add(Conv2D(48, 5, 5, W_regularizer=l2(0.001), activation='elu'))

	#model.add(Dropout(0.50))

	# Add two 3x3 convolution layers (output depth 64, and 64)
	model.add(Conv2D(64, 3, 3, W_regularizer=l2(0.001), activation='elu'))
	#model.add(Conv2D(64, 3, 3, W_regularizer=l2(0.001), activation='elu'))

	# Add a flatten layer
	model.add(Flatten())

	# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
	model.add(Dense(100, W_regularizer=l2(0.001), activation='elu'))
	model.add(Dropout(0.75))
	model.add(Dense(50, W_regularizer=l2(0.001), activation='elu'))
	model.add(Dropout(0.75))
	model.add(Dense(10, W_regularizer=l2(0.001), activation='elu'))
	#model.add(Dropout(0.50))

	# Add a fully connected output layer
	model.add(Dense(1))

	model.compile(optimizer=Adam(lr=1e-4), loss='mse')

	return model


def main():

	batch_size = 128

	x_train, x_val , y_train, y_val = prepare_data()

	train_generator = data_generator(x_train, y_train , batch_size=batch_size)
	validation_generator = data_generator(x_val, y_val, batch_size=batch_size)

	input_shape = (16, 64, 1)
	model = nn_model(input_shape)

	model.fit_generator(train_generator, steps_per_epoch=len(x_train)/batch_size, validation_data=validation_generator,
						validation_steps=len(x_val)/batch_size, nb_epoch=5)

	model.save('model.h5')

if __name__ == '__main__':
	main()


# FIX Revisar configuración de 0.25 para las imagenes de la derecha e izquierda --> 0.275 .. 0.25 .. 0.08 .. 0.2
# Revisar si en udacity se menciona este valor. Revisar si cambiar este valor el tiempo de entrenamiento o el accuracy de conducción

# FIX Revisar otros modelos de red neural para verificar la activiacion de relu en las capas fcc
# FIX Quitar warnings de uso de keras 2.0
# Probar modelo solo con velocidad y asegurar que funciona correctamente
# Agregar velocidad cuando todo este bien