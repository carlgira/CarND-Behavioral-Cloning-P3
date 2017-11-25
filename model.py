import pandas as pd
import numpy as np
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Lambda, Dropout, Cropping2D
from keras.models import Sequential
import keras
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

max_speed = 30.0
max_steering_angle = 25.0


def transform_image(image):
	return image


def prepare_data():
	driving_log = pd.read_csv('run1/driving_log.csv', names=['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed' ])
	num_bins = 20
	steering_angle = driving_log.steering_angle
	steering_angle_hist, steering_angle_bins = np.histogram(steering_angle, num_bins)


	count_bins = []
	for i in range(len(steering_angle_bins)-1):
		count_bins.append(len(driving_log.steering_angle[(driving_log.steering_angle > steering_angle_bins[i]) & (driving_log.steering_angle <= steering_angle_bins[i+1])  ]))

	cut_line = int(np.mean(count_bins)*3)

	for i,v in zip(range(len(count_bins)), count_bins):
		if v > cut_line:
			all_samples = driving_log[(driving_log.steering_angle > steering_angle_bins[i]) & (driving_log.steering_angle <= steering_angle_bins[i+1])  ]
			driving_log_clean = driving_log.drop(all_samples.sample(v - cut_line).index)

	x_data = []
	y_data = []

	for index, row in driving_log_clean.iterrows():
		# Center Image
		x_data.append(row.center_img)
		y_data.append(row.steering_angle)
		#y_data.append([row.speed/max_speed, (row.steering_angle)/max_steering_angle])

		# Left Image
		#x_data.append(row.left_img)
		#y_data.append([row.speed/max_speed, (row.steering_angle + 0.25)/max_steering_angle])
		#y_data.append(row.steering_angle + 0.25)

		# Right Image
		#x_data.append(row.right_img)
		#y_data.append([row.speed/max_speed, (row.steering_angle - 0.25)/max_steering_angle])
		#y_data.append(row.steering_angle - 0.25)

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


def nn_model(input_shape):

	model = Sequential()

	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
	model.add(Cropping2D(cropping=((60,20), (0,0))))
	model.add(Conv2D(24, (5,5), subsample=(2,2), activation='relu'))
	model.add(Conv2D(36, (5,5), subsample=(2,2), activation='relu'))
	model.add(Conv2D(48, (5,5), subsample=(2,2), activation='relu'))
	model.add(Conv2D(64, (3,3), activation='relu'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	return model


def main():

	batch_size = 128

	x_train, x_val , y_train, y_val = prepare_data()

	train_generator = data_generator(x_train, y_train , batch_size=batch_size)
	validation_generator = data_generator(x_val, y_val, batch_size=batch_size)

	input_shape = (160, 320, 3)
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