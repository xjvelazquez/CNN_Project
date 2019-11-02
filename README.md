# CSE154-CNN

# Implementation Instructions
We have provided you with a data loader, a custom PyTorch Dataset class, specifically designed for this dataset,
and a function which partitions the dataset into train, validation splits while creating PyTorch DataLoader object
for each. The latter is an iterator over the elements in each split of the dataset which you will use to retrieve
mini-batches of samples and their labels. Please familiarize yourself with the code and read the comments.

## 1. Evaluating your model:
Before we discuss the neural network details, we must clarify how you will accurately and transparently
evaluate your model’s performance. When dealing with class imbalance, there are cases - particularly in
anomaly detection - where the naive prediction will result in very high overall accuracy. 

## 2. Create a baseline model:
You will create a simple convolutional neural net for the purposes of getting (i) acquainted with PyTorch, (ii)
results to compare with more complex architectures and approaches to solving this multiclass classification
problem. The baseline architecture is the following:
inputs -> conv1 -> conv2 -> conv3 -> conv4 -> fc1 -> fc2 (outputs)
Using the starter code provided, complete the implementation for the architecture as described in the comments, replacing the instances of __ with its corresponding missing value. This includes finishing the __init__() and forward() functions

## 3. Experimentation and your solution:
To get a better sense of how your design choices affect model performance, we encourage you to experiment
with various approaches to solving this multiclass classification problem using a deep CNN. You are welcome
to make a copy of the baseline_cnn.py file and train_model notebook for this purpose (as well as convert
the training code into a .py file).

