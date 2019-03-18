## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)        
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.conv2.weight)     
        nn.init.constant_(self.conv2.bias, 0)                
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)                
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv4_bn = nn.BatchNorm2d(256)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)                
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv5_bn = nn.BatchNorm2d(512)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.constant_(self.conv5.bias, 0)                
        self.pool5 = nn.MaxPool2d(2)

        
        self.fc1 = nn.Linear(12800, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)                
        self.dropout_fc_1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)        
        nn.init.xavier_uniform_(self.fc2.weight) 
        nn.init.constant_(self.fc2.bias, 0)                
        self.dropout_fc_2 = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(512, 136)
        nn.init.xavier_uniform_(self.fc3.weight)  
        nn.init.constant_(self.fc3.bias, 0)                
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x))) # Resulting tensor is of dimensions (N, 32, 111, 111)
        x = self.pool2(self.conv2_bn(F.relu(self.conv2(x)))) # Resulting tensor is of dimensions (N, 64, 55, 55)
#         x = self.pool2(F.relu(self.conv2(x))) # Resulting tensor is of dimensions (N, 64, 55, 55)
        
        x = self.pool3(self.conv3_bn(F.relu(self.conv3(x)))) # Resulting tensor is of dimensions (N, 128, 26, 26)
#         x = self.pool3(F.relu(self.conv3(x))) # Resulting tensor is of dimensions (N, 98, 26, 26)
        
        x = self.pool4(self.conv4_bn(F.relu(self.conv4(x)))) # Resulting tensor is of dimensions (N, 256, 12, 12)
#         x = self.pool4(F.relu(self.conv4(x))) # Resulting tensor is of dimensions (N, 128, 12, 12)
        
        x = self.pool5(self.conv5_bn(F.relu(self.conv5(x)))) # Resulting tensor is of dimensions (N, 512, 5, 5)
#         x = self.pool5(F.relu(self.conv5(x))) # Resulting tensor is of dimensions (N, 128, 5, 5)
        
        # Flatten
        x = x.view(x.size(0), -1) # Flattened layer is of size (N, 12800)
        x = self.dropout_fc_1(F.relu(self.fc1_bn(self.fc1(x)))) # Fully connected layer is of size (N, 1024)
        x = self.dropout_fc_2(F.relu(self.fc2_bn(self.fc2(x)))) # Fully connected layer is of size (N, 512)
        x = self.fc3(x) # Final layer is of size (N, 136)
       
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x