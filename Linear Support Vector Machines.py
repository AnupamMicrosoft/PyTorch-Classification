#Don't change batch size
batch_size = 64
#Hyper-parameters 
input_size = 784  #(dimension of image 28 * 28)
num_classes = 1   #(just -1 and 1 image)
num_epochs = 10  # number of times you will iterate through the full training data
learning_rate = 0.0001 ## step size used by SGD 
momentum = 0.0 ## Momentum is a moving average of our gradients (helps to keep direction)


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
import matplotlib.pyplot as plt

## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data/anupam-data/pytorch/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data/anupam-data/pytorch/data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().view(-1)

train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           sampler=SubsetRandomSampler(subset_indices))

subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().view(-1)

test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                          batch_size=batch_size,
                                          shuffle=False,
                                          sampler=SubsetRandomSampler(subset_indices))

## Displaying some sample images in train_loader with its ground truth

#samples = enumerate(train_loader)
#batch_idx, (sample_data, sample_targets) = next(samples)
#sample_data.shape

#fig = plt.figure()
#for i in range(6):
 #   plt.subplot(2,3,i+1)
  #  plt.tight_layout()
  #  plt.imshow(sample_data[i][0], cmap='gray', interpolation='none')
  #  plt.title("Ground Truth: {}".format(sample_targets[i]))
  #  plt.xticks([])
#    plt.yticks([]) 
#    fig   


#total_step = len(train_loader)
#print(total_step)

class SVM_Loss(nn.modules.Module):    
    def __init__(self):
        super(SVM_Loss,self).__init__()
    def forward(self, outputs, labels):
         return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size

#SVM regression model and Loss
svm_model = nn.Linear(input_size,num_classes)
#model = LogisticRegression(input_size,num_classes)

## Loss criteria and SGD optimizer
svm_loss_criteria = SVM_Loss()

#loss_criteria = nn.CrossEntropyLoss()  

svm_optimizer = torch.optim.SGD(svm_model.parameters(), lr=learning_rate, momentum=momentum)

total_step = len(train_loader)
for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)                      
        labels = Variable(2*(labels.float()-0.5))
                
        # Forward pass        
        outputs = svm_model(images)           
        loss_svm = svm_loss_criteria(outputs, labels)    
        
       
        # Backward and optimize
        svm_optimizer.zero_grad()
        loss_svm.backward()
        svm_optimizer.step()    
        
        #print("Model's parameter after the update:")
        #for param2 in svm_model.parameters():
         #   print(param2)
        total_batches += 1     
        batch_loss += loss_svm.item()

    avg_loss_epoch = batch_loss/total_batches
    print ('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]' 
                   .format(epoch+1, num_epochs, epoch+1, avg_loss_epoch ))
        

        # Test the SVM Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = images.reshape(-1, 28*28)
    
    outputs = svm_model(images)    
    predicted = outputs.data >= 0
    total += labels.size(0) 
    correct += (predicted.view(-1).long() == labels).sum()    
 
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
print("the learning rate is ", learning_rate)
print( "the momentum is", momentum)