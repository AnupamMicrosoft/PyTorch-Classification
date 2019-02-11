# Problem Description
You are asked to implement and solve linear Logistic Regression model and Linear SVM model
(without regularization term) on MNIST dataset. In this task, you only need to perform binary classification
on digit 0 and 1. You need to implement custom loss function, Logistics Loss and Hinge Loss in PyTorch.You
are required to optimize the model by using SGD and Momentum methods. 
# Data
We use MNIST digit classication dataset. Pytorch/torchvision has provide a useful dataloader to automatically
download and load the data into batches. In this homework, we need two class, digit 0 and digit 1, for binary
classication.

# Submit below results
For each of the model, report the average loss for each Batch for each training epoch, where B
is the total number of batches, fb is the model (Logistic regression or Linear SVM) after updated by b-th
batch and Db is the number of data points in b-th batch. An epoch is defined as one iteration of all
dataset. Essentially, during a training epoch, you record down the average training loss of that batch after
you update the model, and then report the average of all such batch-averaged losses after one iteration
of whole dataset. You could plot the results as a figure our simply list down. Please at least report 10
epochs.
 1. Report the final testing accuracy of trained model.
 2. Please compare results for 2 optimizer (SGD and SGD-Momentum)).
 3. Try different step sizes and discuss your findings.
# How to run
Logistics Regression and Support Vector Machine is implemented using PyTorch

1. Python code for Logistics Regression and SVM is provided
2. Open the python files in your favorite IDE
2. Run the file to see the average of batch-averaged loss and the test accuracy
3. Change the learning_rate and momentum hyper parameters on top section of the python file to see different results

# Overview of approach:
I started with reviewing data preparation steps,data was prepared using PyTorch data loader package and split into train and test data set.I printed shape of images, printed few example image to see if the processing is done correctly. During training i implemented custom loss functions for logistics and Linear SVM .For each algorithm, i ran through the training data for every epoch in forward pass we load batches of data, set gradient to zero and calculate the loss using custom loss function.I used PyTorch SGD optimizer to update the parameters using optimizer.step() for different learning rates and momentum till the loss converges. In the last, I used the test data to measure the accuracy of the model. 
