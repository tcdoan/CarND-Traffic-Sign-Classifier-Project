# **Traffic Sign Recognition** 

## Thanh Doan Writeup

---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:
* Load the [German Traffic Sign Benchmark](http://benchmark.ini.rub.de/?section=gtsrb) data set

* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with this written report


---
## Writeup / submission


This writeup, README.md, addressed all [rubric points](https://review.udacity.com/#!/rubrics/481/view) and serves as the project report for submission. The submission includes [Traffic_Sign_Classifier.ipynb](https://github.com/tcdoan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) (project code), this README.md file and [new German traffic sign test images](https://github.com/tcdoan/CarND-Traffic-Sign-Classifier-Project/tree/master/newImages) found on the internet. 


Steps to run *Traffic_Sign_Classifier.ipynb* in your local environment are:
- Clone this repo into your local computer
- Download https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip
- Unzip *traffic-signs-data.zip* and leave *train.p, valid.p, test.p* files in the same directory that house *Traffic_Sign_Classifier.ipynb* notebook
- Create and activate conda environment [car2](https://github.com/tcdoan/CarND-Traffic-Sign-Classifier-Project/blob/master/car2.yml) - this will install all dependency packages such as tensorflow v1.3.0 and opencv v3.4.2.16
    -  conda env create -f car2.yml
    -  activate source car2
- Launch jupyter notebook


## Data Set Summary & Exploration

#### **Basic summary of the data set**

I used `pickle.load(f)` function to load `train.p, valid.p, test.p` files into memory and store training data to `X_train, y_train` numpy arrays. Validation data is stored in `X_valid, y_valid` and test data is stored in `X_test, y_test` arrays. I used simple python and numpy functions to compute basic data summary below:

```python
- X_train shape:  (34799, 32, 32, 3)
- X_valid shape:  ( 4410, 32, 32, 3)
- X_test  shape:  (12630, 32, 32, 3)
- Number of training   examples: 34799
- Number of validation examples:  4410
- Number of testing    examples: 12630
- Image data shape:         (32, 32, 3)
- Number of unique classes/labels:   43
```

#### Exploratory visualization of the dataset.

**First exploratory visualization** is implemented in `visualizeImageClasses(images, labels, title, nrows, ncols)` function. The visualization plots number of training labels for each label class using `matplotlib.pyplot`. The plot shows 43 representative traffic signs, one for each unique label, and label them with *image class names, class IDs, and counts of training examples*.

![Training data: Traffic sign classes and counts](explore1.PNG)

**Second, and third exploratory visualizations** are implemented in `visualizeDataset(images, labels, title='Sample traffic signs', nrows=5, ncols=5)` function. Each visualization randomly selects 25 images and used `matplotlib.pyplot` to plot them. Two dataset visualizations used random traffic signs from training and test sets.

**Third exploratory visualization** is implemented in `visualizeClassCounts(labels, title)` function. The visualization plots the distribution of 43 traffic sign classes from 34,799 training examples. 

**Fourth exploratory visualization** plots the distribution of 43 traffic sign classes using 4,410 data points from validation set. 

![Trafic sign class distribution in training and validation set](explore34.PNG)

**Fifth exploratory visualization** plots the distribution of 43 traffic sign classes using 12,630 data points from test set. 

![Trafic sign class distribution in test set](explore5.PNG)

## Design and Test a Model Architecture

### **Image data preprocessing**

I measured 3 proximate **data normalization** experiments below.

```python
# Experiment 1
# X_train = (X_train - 128.0) / 128.0

# Experiment 2
# X_train = X_train / 255.0 - 0.5

# Experiment 3
X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test =  X_test / 255.0
```
**Why Data normalization:** Data normalization is performed  here to make sure each input pixel has a similar data distribution (0 to 1.0) so that convergence is more reliable and faster while training the neural network. All 3 experiments yield similar results in terms of convergence rate and reliability when trying with learning rate 0.001.

I experimented with converting input images to grayscale images before using them to train the classifier. My experiments did not show considerable imprevements in terms of training error or validation accuracy rate. As the result I decided to skip  'convert to grayscale' step.

**Data augmentation:** Third exploratory visualization showed that some image classes has more training examples than others. For example `class-0` has 180 training examples while `class-2` has 2010 data points for training. To fix this 10-to-1 skewed class distributions I decided to use data augmentation.

Data augmentation is implemented in `augmentData()` function. The function add more images to any class that have less than a `requiredCount` - a threshold that is roughly set to hight class count. New images are generated for augmentation is below.

```python
# For each category that has less than *requiredCount* training images
    # Repeatedly sample random images from existing ones.
       # Until number of training images = *requiredCount* threshold.
```

As I set `requiredCount = 2000` the difference between the original data set and the augmented data set is the following.

```python
X_train.shape before data augmentation (34799, 32, 32, 3)
augment.shape augmented data           (56847, 32, 32, 3)
X_train.shape after data augmentation  (91646, 32, 32, 3)
```

### **Convolutional neural network model architecture**

My initial model architecture, **model-0**,  is a reimplementation of the original LeNet-5 model, the [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791) paper, with modified softmax output layer to emit probabilities for 43 traffic sign classes. With no data augmentation, I trained the model using 34,799 RBG images of size (32, 32, 3). After 30 epochs of training the training accuracy is 100%. The validation accuracy is 91.6% and test accuracy is 90.6%.

![learning curve showing training and validation accuracy graph](model0.PNG)

The precision, recall and f1-score classification metrics on `never-seen-before test set` for **model-0** are:

![precision, recall and f1-score classification metrics for model0](m0Metrics.PNG)

![precision, recall and f1-score definition](metrics.PNG)

Next, I train **model-0** with data augmentation. Using 91,646 augmented images of size (32, 32, 3) I trained the model for 30 epochs and saw the training accuracy is 100%, the validation accuracy is 93.3% and the test accuracy is 92.1%.

Next **model-1** is derived from model-0 by adding **dropout** regularization in-front of rectified linear activation function components for 2 fully connected layers `fc1` and `fc2`. 

```python
# Fully Connected. Input: 5x5xK2. Output: FCK1
fc1 = tf.reshape(pool2, [-1, 5*5*K2])
wd1 = tf.Variable(tf.truncated_normal([5*5*K2, FCK1], mean=mu, stddev=sigma))
bd1 = tf.Variable(tf.zeros([FCK1]))
fc1 = tf.add(tf.matmul(fc1, wd1), bd1)

# adding DROPOUT regularization 
fc1 = tf.nn.dropout(fc1, self.keep_prob)
fc1 = tf.nn.relu(fc1)

# Fully Connected. Input: FCK1. Output: FCK2
wd2 = tf.Variable(tf.truncated_normal([FCK1, FCK2], mean=mu, stddev=sigma))
bd2 = tf.Variable(tf.zeros([FCK2]))
fc2 = tf.add(tf.matmul(fc1, wd2), bd2)

# Adding DROPOUT regularization 
fc2 = tf.nn.dropout(fc2, self.keep_prob)
fc2 = tf.nn.relu(fc2)
```
I trained **model-1** using 91,646 augmented images for 30 epochs. The training accuracy is 99.8%. The validation accuracy is 95.9% and test accuracy is 93.9%. The weighted average precision, recall and f1-score classification metrics on the testset is 0.94, 0.94 and 0.94. However for **class 27** precision is a lowly 0.48, recall is 0.53 and f-measure is only 0.51.

Next **model-2** is derived from model-1 by adding more filters to the 2 convolutional layers. In the first convolutional layer, `conv1`, I increased the number of filters from **6 to 38**.  In the second convolutional layer, `conv2`, I increased the number of filters from **16 to 64**.

**Model-2**, the final model, has the following layers:

![Model-2 architecture](modelArchitecture.PNG)

I trained **model-2** using 91,646 augmented images for 30 epochs. The training accuracy is 100%. The validation accuracy is 97.9% and test accuracy is 96.69%. The learning curves as the results of training **model-2** using augmented data is below. 

![Model-2 learning curve](m2curve.png)

To train the model. I set up the loss function and the training operation as below

```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels=one_hot_y))
training = tf.train.AdamOptimizer(learning_ratee).minimize(loss)
```

The loss function is 
    $ loss = \frac{1} {2}  $

the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

![Model-2 learning curve](m2train.png)


To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

