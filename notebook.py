
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall
from tqdm import tqdm


# Load datasets
from torchvision import datasets
import torchvision.transforms as transforms

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


import matplotlib.pyplot as plt

image, label = next(iter(train_data))

image.shape

num_classes = len(train_data.classes)
print(num_classes)

image_size = train_data[0][0].shape
image_size

train_data[0][0]

 
def view_first_10_images(dataset):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        image, label = dataset[i]
        axes[i//5, i%5].imshow(image.squeeze(), cmap='gray')
        axes[i//5, i%5].set_title(f'Label: {label} - {dataset.classes[label]}')
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    plt.show()

view_first_10_images(train_data)

 
len(train_data), len(test_data)

   
# above is a few sample of clothings of some lables

 
plt.imshow(image.squeeze())
plt.title(f'{label} - {train_data.classes[label]}')

 
type(label)

   
# model development stage
# 
# The Net class is a simple CNN that consists of two convolutional layers followed by max pooling, a flattening layer, and a fully connected layer for classification. The forward method defines how the input data passes through these layers to produce output class probabilities.

 
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(16* 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.classifier(x)
        return x

model = Net(num_classes)

 
model

   
# initializing the optimizer, loss function

 

from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss()

 
'''def run_model(data_file):
    rate = [0.001, 0.005, 0.01, 0.05, 0.1]
    for lr in rate:
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = .0004)
        losses = []
        for epoch in tqdm(range(5)):
            for image, label in data_file:
                label, image = torch.tensor(label), torch.tensor(image)
                #label = label.unsqueeze(0)
                optimizer.zero_grad()
                #forward pass
                outputs = model(image)
                loss = criterion(outputs, label)
                #back pass
                loss.backward()
                #update parameters
                optimizer.step()
                losses.append(loss.item())

                
            print(f'the loss for epoch {epoch+1}: with {lr} = {loss.item():.4f}')'''

 
def run_model(data_file, lr=0.001, epochs=5):
    model = Net(num_classes)  # Reinitialize model for fresh training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0004)
    losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for image, label in data_file:
            # No need to convert to tensor again; DataLoader already provides tensors
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(data_file)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        return model, losses

 
datafor_train = DataLoader(train_data, batch_size= 8, shuffle=True)

 
trained_model, training_losses = run_model(datafor_train, lr=0.001, epochs=5)

   
# the learning rate betweeen 0.001 and 0.005 will mostly yield the least losses

 
'''model.train()
for lr in [0.02, 0.04, 0.06, 0.08, 0.1]:
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = .0004)
    for epoch in tqdm(range(2)):
        model_x = tqdm(run_model(datafor_train))'''

   
# Testing the model on the test data

 
model.eval()

 
datafor_test = DataLoader(test_data, batch_size = 8, shuffle=False)

 
image, label = next(iter(test_data))

 
from torchmetrics import Accuracy, Recall, Precision

 
acc_metric = Accuracy(task= 'multiclass', num_classes= num_classes)
prec_metric = Precision(task = 'multiclass', num_classes= num_classes, average= 'macro')
recall_metric = Recall(task = 'multiclass', num_classes=num_classes)

 
def measure_metrics(model, test_file):
    predictions = []

    acc_metric.reset()
    prec_metric.reset()
    recall_metric.reset()

    model.eval()
    with torch.no_grad():
        for image, label in tqdm(test_file):
            outputs = model(image)
            predicted = torch.argmax(outputs, dim=1)
            predictions.append(predicted)
            acc_metric(predicted, label)
            prec_metric(predicted, label)
            recall_metric(predicted, label)
            
    accuracy = acc_metric.compute().item()
    precision = prec_metric.compute().item()
    recall = recall_metric.compute().item()
    return accuracy, precision, recall, predictions

 
accuracy, precision, recall, predictions = measure_metrics(trained_model, datafor_test)

 
print(f'accuracy: {accuracy}, \n precision: {precision}, \n recall: {recall}')

 
model_dummy_input = torch.randn(8, 1, 28, 28)

 
torch.onnx.export(trained_model, model_dummy_input, "clothing_classifier.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output': {0: 'batch_size'}})

 



