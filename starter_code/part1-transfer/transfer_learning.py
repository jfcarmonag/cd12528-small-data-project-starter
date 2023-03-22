# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models

from collections import OrderedDict



# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

#<<<YOUR CODE HERE>>>

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

#<<<YOUR CODE HERE>>>

#hint, create a variable that contains the class_names. You can get them from the ImageFolder

data_dir = 'imagedata-50'
image_datasets = {
    'train': datasets.ImageFolder(data_dir + '/train', data_transforms['train']),
    'valid': datasets.ImageFolder(data_dir + '/val', data_transforms['valid']),
    'test': datasets.ImageFolder(data_dir + '/test', data_transforms['test']),
}

train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)

class_names = {idx: cl for idx, cl in image_datasets['train'].class_to_idx.items()}

# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

#<<<YOUR CODE HERE>>>
model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1000)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(1000, 3)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

#<<<YOUR CODE HERE>>>
num_epochs = 2
criterion = nn.NLLLoss()
lr=0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
   trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
   test_model(test_loader, trained_model, class_names)

if __name__ == '__main__':
    main()
    print("done")