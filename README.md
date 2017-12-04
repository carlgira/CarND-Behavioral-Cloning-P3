# **Behavioral Cloning** 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* **README.md** writeup of project summarizing the results
* **Data-Exploration-Preprocessing**, notebook with some notes on data exploration and image preprocesing
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **track1.mp4** (a video recording of the vehicle driving autonomously around track1, **one full lap**)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

At first i used the NVIDIA architecture, my idea was to create training data using both tracks and train the neural network with that data. But i could not make the second track work, i tried several things like change dropout rate, adding more data, augment data, change velocity on the simulator, change the amount of data points that should be removed with steering of 0.0, modifying image preprocessing like resizing, cropping, or change color space. Finally i simply couldn't make it work.

So, i tried to focus on the first track, but it was easy to train (well, actually after figuring out that the cv2 load the image in BGR and the simulator loaded the image in RGB). The NVIDIA model was not necessary, just to big for that simple task. Finally i decided to simplify the architecture, so i just start removing layers and in the end i only leave two convolutional layers, one fully connected layer and a dropout layer.

#### 2. Attempts to reduce overfitting in the model

I did three things to try avoid overfiting:

1. **Create a well balanced dataset,** The track 1 has lots of parts where the vehicle is only going totally straight, so the amount of samples with "steering angle" of 0.0 value, was a lot bigger that the all other steering points. The dataset was preprocessed so the amount of images of the different values of the steering angle where much better distributed, removing a good number of the samples that were not needed (this is better explained in section **Creation of the Training Set & Training Process***), in the model.py file in the method **prepare_data** line 26
2. **Using a dropout layer,** with a keeping probability of 0.5, in the model.py file in the method **nn_model** line 114
3. **Training the model on small number of epochs,** check the training and validation loss and make sure that the validation loss dont increase or jumps forward and backwards. The number of epochs was left at **3**, in the model.py file in the method **main** line 138

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I think the most important parameter that i had to tune was the "amount" of data that must be "removed" with the steering angle of 0.0. 

- If the filter was to high, and a greater proportion of 0.0 steering values where on the dataset, the car was **really well centered** on the road, but (of course) the curves where not well taken.
- If the filter was to low, and a lower proportion of 0.0 steering values where on the dataset, the **car stays on the road** but the car begins to move from one side to the other on the road.

The distribution of data can be seen in the **Data-Exploration-Preprocessing** notebook and in the model.py file in the method **prepare_data** line 26 ) 

The model used an adam optimizer, an the learning rate used was **0.0001**

The dropout probability used was **0.5**

#### 4. Appropriate training data

Did two recordings of the first track, in forward and backward direction, making three laps with each one. With track 2, two laps forward and two laps in backward direction.

Also i use shared data from other udacity student.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As mentioned before i started to use the NVIDIA architecture, but seeing that i could not make work track2, i decided to simplify the model for the track1 that in fact it can be trained with much smaller model.

I started to test removing layers from the original architecture and ended with a smaller model that is working well. I also decided to preprocess the images so, i resize and crop the data 

I separate the data in a train and validation datasets using a 80% of data in train and 20% for validation.

I had to tune the amount of steering values at 0.0 to be removed (as commented bellow) and change the number of epochs to make sure that the models was not overffiting.

Had to play with the number of epochs to make sure that the model was not overfitting

The model was tested on track1 and it was working correctly. In the second track, the vehicle simply starts with a high turn to the left and even seeing that when the vehicle sees the barrier and tries to correct the course turning to the right, is too late and the car get stuck.

#### 2. Final Model Architecture

The simplified architecture is as follows: (in the model.py file in the method **nn_model** line 102)

- Normalization layer, (a lambda layer)
- Convolutional layer, 32 filters stride of (3,3) and 'elu' activation to introduce nonlinearity
- MaxPool layer, pool size (4, 4) and stride of (4, 4)
- Convolutional layer, 64 filters stride of (3,3) and 'elu' activation to introduce nonlinearity
- Flatten layer
- Droupout layer, keeping probability of 0.5
- Fully connected layer, 50 units and 'elu' activation to introduce nonlinearity
- Fully connected layer, 1 unit

#### 3. Creation of the Training Set & Training Process

I think one of hardest thing to do with this project was to have a good dataset, the model, was not "that important". You have to make sure of two things, a sufficient number of samples, and have a well distributed dataset.

For that, you have to make sure to make several laps, forward and backward direction in both tracks. The forward and backward direction was important to make sure that you have a good number of samples with right and left steering angles, as **track 1 was mostly and oval**.

I got 1GB of training data, but after cleaning that data of steering values of 0.0 you kind of remove 70% of samples. I went to slack and other udacity student was sharing 3GB of their own dataset that i was very happy to use. (i'm was a bad driver)

After get this amount of data, i decided in the final model not to augment the data (flip the images or change brightness for example). But decided to preprocess the image, as seeing in other similar projects.

The final image, is transformed to HSV color space, resized to 32x64 and use a area of interest to create images of only 16x64.

The images where shuffled and separated in two sets for training and validation.

# Conclusions

- Probably the hardest part of the project was to create a good distributed dataset. It was necessary to create a well balanced steering samples, so the steering 0.0 was equally represented as the other values

- One important variable to tune was the rate of steering 0.0 value to be removed. If the cut line was not equilibrated (to low or to high), the training of the vehicle was not correct.

- The simplified neural network model was enough for the simulation on the track1. Cant be sure why the track2 it was not working, i think it was necessary more samples.
