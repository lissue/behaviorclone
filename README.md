# **Behavioral Cloning**

## Goals

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./center_1.jpg "Grayscaling"
[image3]: ./correction-1.jpg "Recovery Image"
[image4]: ./correction-2.jpg "Recovery Image"
[image5]: ./correction-3.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Key files to be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Environment

The following packages need to be installed to run `drive.py`.

```bash
pip install python-socketio eventlet flask h5py
```

PyAV is the easiest way to install the x264 codec to run `video.py`.

```bash
conda install av -c conda-forge
```

There is some versions issue among Python, Keras, and Tensorflow. Keras and TensorFlow need to be updated to the latest version, otherwise there may be "unknown opcode" or "bad marshal data" errors when a lambda layer is used to normalize the images. And this seems to only work for Python 3.5 -- once I update Python to 3.6 the problem comes back even if both Keras and TensorFlow were updated.

```bash
pip install tensorflow --upgrade
pip install keras --upgrade
```

In sum, I used Python 3.5.4, TensorFlow 1.4.1, and Keras 2.1.2 for both training of the model on AWS and running the simulation on my local machine.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Model Architecture and Code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Overview

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filters sizes and depths between 24 and 64 (model.py [lines 69-80](model.py#L69-L80))

The model includes RELU layers to introduce nonlinearity ([lines 70-72](model.py#L70-L72)), and the data is normalized and cropped in the model using a Keras lambda layer ([code line 67](model.py#L67)).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py [lines 69](model.py#L69)[&77](model.py#L77)).

The model was trained and validated on different data sets to ensure that the model was not overfitting ([line 22](model.py#L22)). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py [line 82](model.py#L82)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, as well as one lap driving in reverse direction of the track.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to utilize open source architectures that are known to work in dealing with similar problems, and tweak to ensure the implementation works for the project.

My first step was to use a convolution neural network model similar to the [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). I thought this model might be appropriate because of its great performance in the ImageNet Challenge. But then I decided to use the [Nvidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) as it was directly used on self-driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I included dropouts.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such as the left turn leading to the bridge. To improve the driving behavior in these cases, I added more of the manual driving at the locations to the training and validation datasets.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py [lines 69-80](model.py#L69-L80)) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 17504 data points. I then preprocessed this data by cropping the top 50 rows and bottom 20 rows of the image, so that the view is always focused on the road itself.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the monotonic reduction in MSE. I used an adam optimizer so that manually training the learning rate wasn't necessary.
