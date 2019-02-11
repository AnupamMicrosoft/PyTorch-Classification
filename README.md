# PyTorch-Classification
Logistics Regression and Support Vector Machine using PyTorch
How to run
1. Python File for Logistics Regression and SVM is provide 
2. Open the python files in your favorite IDE
2. Run the file to see the average of batch-averaged loss and the test accuracy
3. Change the learning_rate and momentum hyper parameters on top section of the python file to see different results

Overview of my code:
I started with reviewing data preparation steps,data was prepared using PyTorch data loader package and split into train and test data set.
I printed shape of images, printed few example image to see if the processing is done correctly. 
During training i implemented custom loss functions for logistics and Linear SVM as taught in the class. 
For each algorithm, i ran through the training data for every epoch in forward pass we load batches of data, set gradient to zero and calculate the loss using custom loss function.
I used PyTorch SGD optimizer to update the parameters using optimizer.step() for different learning rates and momentum till the loss converges. 
In the last, I used the test data to measure the accuracy of the model. 
