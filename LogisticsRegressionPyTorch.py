#Don't change batch size
batch_size = 64
#Hyper-parameters 
input_size = 784  #(dimension of image 28 * 28)
num_classes = 1   #(just -1 and 1 image)
num_epochs = 10  # number of times you will iterate through the full training data
learning_rate = 0.0001 ## step size used by SGD 
momentum = 0.9 ## Momentum is a moving average of our gradients (helps to keep direction)


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

class Regress_Loss(nn.modules.Module):    
    def __init__(self):
        super(Regress_Loss,self).__init__()
    def forward(self, outputs, labels):
        batch_size = outputs.size()[0]
        return torch.sum(torch.log(1 + torch.exp(-(outputs.t()*labels))))/batch_size


#Logistic regression model and Loss
logistics_model = 0
logistics_model = nn.Linear(input_size,num_classes)

## Custom Loss criteria and SGD optimizer
loss_criteria = Regress_Loss()

optimizer = torch.optim.SGD(logistics_model.parameters(), lr=learning_rate, momentum= momentum)
total_step = len(train_loader)

## Train the model parameters

for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)            
        labels = Variable(2*(labels.float()-0.5))
        
        outputs = logistics_model(images)               
        loss = loss_criteria(outputs, labels)    
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        total_batches += 1     
        batch_loss += loss.item()

    avg_loss_epoch = batch_loss/total_batches
    print ('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]' 
                   .format(epoch+1, num_epochs, epoch+1, avg_loss_epoch ))


## Test the model

# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = images.reshape(-1, 28*28)

    outputs_test = torch.sigmoid(logistics_model(images))
    predicted = outputs_test.data >= 0.5 
 
    total += labels.size(0) 
    
    correct += (predicted.view(-1).long() == labels).sum()
    
   
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
#print("number of total test images", total)
#print
#print("numbers of correctly predicted test images" ,correct )
print("the learning rate is ", learning_rate)
print( "the momentum is", momentum)
        