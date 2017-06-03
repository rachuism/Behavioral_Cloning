#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Crop the image"
[image2]: ./examples/placeholder.png "Gamma"
[image3]: ./examples/placeholder_small.png "Resize"
[image4]: ./examples/placeholder_small.png "Mirror"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I've studied the Nvidia model for making an aproximation to the perfect network. It takes 5 Convolutional layers with kernel sizes of 5x5, 5x5 5x5, 3x3, 3x3 and depths of 24, 36, 48 and 64. Then a flatten layer arrange all the data into one line and then I use some Dense layers with 1164, 100, 50, 10 and 1 neuron. After each convolutional layer I implement a Pooling layer for decreasing the number of characteristics and and Relu layer for introducing some non-linearities.

At the beginning of the network, I use a lambda layer for normalizing all the data before I pass it trough the network. 

 
####2. Attempts to reduce overfitting in the model

As I wanted to make my model as similar as possible to the Nvidia model I decided to not use any Dropout layer.

My model takes from the data set around 8000 examples but for preventing overfitting I've experimented reducing gradually the number of samples. Then my training validation set only contains 1000 examples "samples_per_epoch=1000" that is the optimumu number. Also my model only needs 3 epochs for training itself without overfitting as the loss is very small since the beginning.

####3. Model parameter tuning

My model uses an adam optimizer. It consists on an adaptative learning rate for each parameter so I didn't need to tune it manually.

####4. Appropriate training data

I tried to use my own training data but it wasn't accurate enought for training the model. Finally I decided to use the data provided as it performs better when setting the weights. In the next section I explain it deeply.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I made a research before stablishing my model. Nvidia had already made a study for creating their own model. For achieving it they compared the human behaviour and the ouput (steering angle) from the model. They trained the weights of the network to minimize the mean square error between the human steering angle and the ouput from the network. As I discovered that the Nvidia model fitted perfectly for this project I decided to use it. 

For training the network I first have to choose the data images to use. I pick the images from the dataset and divide them in training samples and validation samples, "train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)". It takes a 80% for the training samples and 20% for the validation samples. 

Then I run the simulator for checking if all that processes were enough. The car still had some doubts like turning right after the bridge, sometimes it took the dirt road and  eventually crashed with a rock. That's why I decided to make some preprocessing before training the model (explained in point 3). 

As I reduced the number of samples per epoch I realised that it performed better 
Then I run the simulator and it completed the whole lap without crashing in any obstacle.

####2. Final Model Architecture

The network consists of 9 layers, including a normalizing layer, 5 convolutional layers and 3 fully connected layers. I input an image that is splitted into YUV planes (a color space).
The first layer of the network normalize the data. It is and advantage normalizing the data inside the network because this way it can be accelerated via GPU and the scheme can be altered with the network architecture.

Then the convolutional layers extract the features. There are 3 convolutional layers with kernel size 3x3 and two with kernel size 5x5. 
The Flatten layer is the first requirement for the fully connected layers. The fully connected layers are in charge of controlling the steering angle. The final architecture is as follows:

3@50x50 Input plane
3@50x50 Normalized input plane
Convolution, depth=24, kernel=5x5
Convolution, depth=36, kernel=5x5 
Convolution, depth=48, kernel=5x5
Convolution, depth=64, kernel=3x3
Convolution, depth=64, kernel=3x3
Flatten
Dense, 1164 neurons
Dense, 100 neurons
Dense, 50 neurons
Dense, 10 neurons
Dense, 1 neuron

Total params: 62,905,719
Trainable params: 62,905,719
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

I used the dataset given by Udacity. As I need two subsets I divide all the frames in data for training and for validating. I choose that a 20% of the data belongs to validation samples and the rest belongs to training samples. 

I was taught along the course a technique for spliting the data in some batches so that the GPU doesn't run out of memory, it is called "Generator". A good batch size should be around 128 samples as maximum. I've decided to choose 50.

During the generator I create two arrays that will contain all the images and all the angles that are going to be trained. For transforming the data I used some preprocessing. 
First I crop the image so that only the area from the road can be analized. The result is:

![alt text][image1]

Then I apply gamma correction. First, our image pixel intensities must be scaled from the range [0, 255] to [0, 1.0]. From there, we obtain our output gamma corrected image by applying the following equation:

O = I ^ (1 / G)

Where I is our input image and G is our gamma value. The output image O is then scaled back to the range [0, 255].

Gamma values < 1 will shift the image towards the darker end of the spectrum while gamma values > 1 will make the image appear lighter. A gamma value of G=1 will have no affect on the input image:

![alt text][image2]

Finally I resize the image to the size of the input of the pipeline, in my case a size of 50x50:

![alt text][image3]

I also tried flipping the image but it wasn't a good approach for the performance of my model:

![alt text][image4]

Now here is the generator point, it yields in this position and the batch data goes through the network until the next batch. I tried using all the samples from the data set but the simulator reveals that with just around 20% of the samples it works the best. 

I decided to use an Adam optimizer as I explained before so I didn't choose any learning rate.
