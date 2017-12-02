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

max_steering_angle = 25.0

def transform_image(img):
	#img_transform = cv2.resize((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),(img_cols,img_rows))[12:img_cols,]

	#resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (img_cols,img_rows))[12:img_rows-4,]
	#return np.reshape(resized, (16, img_cols, 3))
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prepare_data():
	#files = ['run1/driving_log.csv', 'run3/driving_log.csv']
	files = ['my_data/driving_log.csv']

	driving_log = pd.read_csv(files[0], names=['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed' ])

	for i in range(1, len(files)):
		driving_log = driving_log.append(pd.read_csv(files[i], names=['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed' ]))

	num_bins = 20
	steering_angle = driving_log.steering_angle
	steering_angle_hist, steering_angle_bins = np.histogram(steering_angle, num_bins)

	count_bins = []
	for i in range(len(steering_angle_bins)-1):
		count_bins.append(len(driving_log.steering_angle[(driving_log.steering_angle > steering_angle_bins[i]) & (driving_log.steering_angle <= steering_angle_bins[i+1])  ]))

	cut_line = int(np.mean(count_bins)/2)

	for i in range(len(count_bins)):
		bin_samples = driving_log[(driving_log.steering_angle > steering_angle_bins[i]) & (driving_log.steering_angle <= steering_angle_bins[i+1])  ]
		if len(bin_samples) > cut_line:
			driving_log_clean = driving_log.drop(bin_samples.sample(len(bin_samples) - cut_line).index)

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
		y_data.append((row.steering_angle + 0.15))

		# Right Image
		x_data.append(row.right_img)
		#y_data.append([row.speed/max_speed, (row.steering_angle - 0.25)/max_steering_angle])
		y_data.append((row.steering_angle - 0.15))

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

	model = Sequential([
		# Normalize image to -1.0 to 1.0
		Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape),
		Cropping2D(((60,20), (0,0))),
		# Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation
		Conv2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .1 (keep probability of .9)
		Dropout(.1),
		# Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
		Conv2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .2 (keep probability of .8)
		Dropout(.2),
		# Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
		Conv2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
		# Flatten
		Flatten(),
		# Dropout with drop probability of .3 (keep probability of .7)
		Dropout(.3),
		# Fully-connected layer 1 | 100 neurons | elu activation
		Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .5
		Dropout(.5),
		# Fully-connected layer 2 | 50 neurons | elu activation
		Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .5
		Dropout(.5),
		# Fully-connected layer 3 | 10 neurons | elu activation
		Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
		# Dropout with drop probability of .5
		Dropout(.5),
		# Output
		Dense(1, activation='linear', init='he_normal')
	])

	model.compile(optimizer=Adam(lr=1e-4), loss='mse')

	return model


def main():

	batch_size = 128

	x_train, x_val , y_train, y_val = prepare_data()

	train_generator = data_generator(x_train, y_train , batch_size=batch_size)
	validation_generator = data_generator(x_val, y_val, batch_size=batch_size)

	input_shape = (160, 320, 3)
	model = nn_model(input_shape)

	model.fit_generator(train_generator, steps_per_epoch=len(x_train)/batch_size, validation_data=validation_generator,
						validation_steps=len(x_val)/batch_size, nb_epoch=2)

	model.save('model.h5')

if __name__ == '__main__':
	main()


# FIX Revisar configuración de 0.25 para las imagenes de la derecha e izquierda --> 0.275 .. 0.25 .. 0.08 .. 0.2
# Revisar si en udacity se menciona este valor. Revisar si cambiar este valor el tiempo de entrenamiento o el accuracy de conducción

# FIX Revisar otros modelos de red neural para verificar la activiacion de relu en las capas fcc
# FIX Quitar warnings de uso de keras 2.0
# Probar modelo solo con velocidad y asegurar que funciona correctamente
# Agregar velocidad cuando todo este bien