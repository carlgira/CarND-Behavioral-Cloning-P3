# **Behavioral Cloning** 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **README.md** writeup of project summarizing the results
* **Data-Exploration-Preprocessing**, notebook with some notes on data exploration and preprocesing
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

At first i used the NVIDIA architecture, my idea was to create training data using both tracks and train the neural network with that data. But i could not make the second track work, i tried several things like removing layers, change dropout rate, adding more data, augment data, change velocity on the simulator, change the amount of data points that should be removed with steering of 0.0, modifying image preprosecing like resizing, crpoping, or change color space. I simply couldn't make it work.

So, i tried to focus on the first track, but is was easy to train (well, actually after figuring out that the cv2 load the image in BGR and the simulator load the image in RGB). The NVIDIA model was not necesary, it simply was a waste in procesing. Finnaly i decided to simplyfy the architecture so i just start removing layers and in the end i only leave two convolutional layers (with just one it wasn't working for me), one fully connected layer and a dropout layer.


#### 2. Attempts to reduce overfitting in the model

I did three things to try no to overfit the model:

1. **Create a good distribution of the data,** The track 1 has lots of parts where the vehicle is only going totally straigh, so the amount of "steering angle" with 0.0 value, was a lot bigger that the all other steering points. The dataset was preprocesed so the amount of images of the different values of the steering angle where much better distributed, removing a good number of the samples that were not needed (this is better explained in section **Creation of the Training Set & Training Process***), in the model.py file in the method **prepare_data** line 26
2. **Using a dropout layer,** with a keeping probability of 0.5, in the model.py file in the method **nn_model** line 114
3. **Training the model on small number of epochs,** check the training and validation loss and make sure that the validation loss dont increase or jumps foward and backwards. The number of epochs was left at **3**, in the model.py file in the method **main** line 138

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I think the most important parameter that i had to tune was the "amount" of data that must be "removed" with the steering angle of 0.0. 

- If the filter was to high, and a greater proportion of 0.0 steering values where on the dataset, the car was **really well centered** on the half of the road, but (of course) the curves where not well taken.
- If the filter was to low, and a lower proportion of 0.0 steering values where on the dataset, the **car stays on the road** but the car begins to move from one side to the other on the road.

So after lots of test, i trying to create half way between the two main principles, (car stays in the road well centered), the idea was to separate the values of steering angle in equally separated bins, and make sure that all bins had **at maximum** the mean size of all bins. (the distribution of data can be seen in the **Data-Exploration-Preprocessing** notebook and in the model.py file in the method **prepare_data** line 26 ) 

The model used an adam optimizer, an the learning rate used was **0.0001**

The dropout probability used was **0.5**

#### 4. Appropriate training data

Did two recordings of the first track, one foward and the other in backward direction, making three laps with each one, with track number two two laps foward and two laps in backward direction.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As mentioned before i started to use the NVIDIA architecture, but seeing that i could not make work track2, i decided to simplify the model for the track1 that in fact it can be trained with much smaller model.

I started to test removing layers from the original architecture and ended with a smaller model that is working well. I also decided to preprocess the images so, i resize and crop the data 

I separate the data in a train and validation datasets with using a 80% of data in train and 20% for validation.

I had to tune the amount of steering values at 0.0 to be removed (as commeted bellow) and change the number of epochs to make sure that the models was not overffiting.

Had to play with the number of epochs to make sure that the model was not overfitting

The model was tested on track1 and it was working correctly. In the second track, the vehicle simply starts with a high turn to the left and even seening that when the vehicle sees the barrier and tries to correct the course turning to the right, is too late and the car get stuck.

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

I think one of hardest thing to do with this project was to have a good dataset, the model, was not "that important". You have to make sure of two things, a sufficent number of samples, and have a well distributed dataset.

For that, you have to make sure to make several laps, foward and backward direction in both tracks. The foward and backward direction was important to have to make sure that you have a good number of samples with right and left steering angles, as track 1 was mostly and oval.

I got 1GB of training data, but after cleaning that data of steering values of 0.0 you kind of remove 70% of samples. I went to slack and other udacity student was sharing 3GB of their own dataset that i was very happy to use. (i'm was a very bad driver)

After get this amount of data, i decided in the final model not to augment the data (flip the images or change brightness for example). But decided to preprocess the image, as seeing in other student projects on github.

The final image, is trasnformed to HSV color space, resized to 32x64 and use a area of interest to create images of only 16x64.

The images where shuffled and separated in two sets for training and validation.

# Conclusions

- The data extrac
