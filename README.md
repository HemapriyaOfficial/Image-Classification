# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
Design and implement a Convolutional Neural Network (CNN) to classify grayscale images from the FashionMNIST dataset into 10 distinct categories. The model should learn to recognize patterns and features in the images to accurately predict their respective classes.


## Neural Network Model

![image](https://github.com/user-attachments/assets/adcff8b3-b709-46b8-9acb-f3d956f15772)


## DESIGN STEPS

STEP 1:
Classify grayscale images into 10 categories using a CNN.

STEP 2: 
Load the FashionMNIST dataset with 60,000 training and 10,000 test images.

STEP 3:
Convert images to tensors, normalize, and create DataLoaders for efficient processing.

STEP 4: 
Build a CNN with convolution, activation, pooling, and fully connected layers.

STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer over multiple epochs.

STEP 6: 
Test the model, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 7:
Predict new images and display actual vs. predicted labels for visual analysis.

## PROGRAM

### Name: HEMAPRIYA K
### Register Number: 212223040066
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x): 
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch

![image](https://github.com/user-attachments/assets/c69f21cf-df5f-48a7-924e-f8f46c091a8b)


### Confusion Matrix

![image](https://github.com/user-attachments/assets/0fa5feba-bdc3-4077-afa9-693b19328509)


### Classification Report

![image](https://github.com/user-attachments/assets/3be308c1-9db2-47c9-8cd7-52a1dd189ac3)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/efb1bf60-1e9a-4407-97df-ddb883dc9e79)


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
