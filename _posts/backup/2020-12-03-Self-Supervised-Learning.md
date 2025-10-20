---
layout: post
title: Self Supervised Learning
github: https://github.com/BenoitLeguay/Self-Supervised-Learning
---

In this section, we will show the efficiency of Self Supervised Learning with a very simple example.

The goal here is to classify cats and dogs. Yet we don't have enough labeled data to used a classical supervised learning algorithm. Labeling data can be very long and expensive. 
Then we decide to use **Self Supervised Learning**.

<br/>

We take our whole dataset and perform a 90° and a 180° rotation on every images. Now we have a dataset with labels (90 rot and 180 rot). We can train a Convolutional Neural Network to predict the rotation degree. 
As soon as the network is sufficiently trained we can go back to our dogs and cats classification task.
We are going to use our pre-trained rotation network to make our cats and dogs classification. Since the network has already been trained on detecting edges and other cats and dogs components (nose, eyes, etc...) the classification should be more easy and thus require less labeled data.

*nb: the goal here is not to achieve the best score on cats and dogs dataset but to prove self supervised learning promise: good performance with less labeled data.*

<br/><br/>

## 1. ROTATION CLASSIFICATION


![png]({{ site.baseurl }}/images/Self%20Supervised%20Learning_8_0.png)


Here, the goal of the classifier is to determine if a 90 degrees rotation or a 180 degrees rotation has been applied to the image. **The labeling task - for the rotation - is free** and our classifier will train on our data. Thus It will train on detecting edges and structures among our datasets. Note that, the idea of cats and dogs are not important here.

![png]({{ site.baseurl }}/images/Self%20Supervised%20Learning_16_0.png)

The network succeed on what it has been trained on: classify rotation degree. 

<br/><br/>


## 2. Cats and Dogs Classification

We will use an embedded layer from the Rotate Classifier we just trained. As a baseline we will train the same network architecture on our dataset. On the pre-trained classifier, **only 50%** of the training set will be used. 

Thus, we are going to observe and compare the learning process between a pre-trained neural network fed with half of the dataset and a new neural network which will receive the whole dataset. 

![png]({{ site.baseurl }}/images/Self%20Supervised%20Learning_27_0.png)

<br/>


### 2.2 Training


![png]({{ site.baseurl }}/images/Self%20Supervised%20Learning_32_0.png)

Our classifier with pre trained weights (on rotation task) performs better that the other one. **We got here a better result with a smaller amount of labeled data.** 

<br/>

### Test set results

Despite being trained on less data our classifier with pre trained weights performs slightly better than the one that used 2 times labeled data.

<br/>

    self supervised learning test set acc: 0.7062600255012512
    baseline test set acc: 0.6950240731239319
    self supervised learning
    tensor([[257.,  51.],
            [132., 183.]])
    baseline
    tensor([[225.,  83.],
            [107., 208.]])