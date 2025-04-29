
#Second Model: Moderate Dementia vs Signs of mild Dementia
#getting the data for the first model ready using DataLoader
#add a few images for moderate but changing brightness, give moderate dementia more data points
import os
from torchvision import transforms
from PIL import Image
#this will give a random brightness change
brightness_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.10),
])

input_dir = "Processed_MRI_Data_For_Second_Model"
input_sub_dir = "Signs_of_Moderate_Dimentia"
output_path = os.path.join(input_dir, input_sub_dir)

for mri_image in os.listdir(output_path):
    img_path = os.path.join(output_path, mri_image)
    img = Image.open(img_path)  # Load image
    changed_img1 = brightness_transform(img)
    changed_img1.save(os.path.join(output_path, f"{mri_image}_changed_brightness_1.jpg"))
    changed_img2 = brightness_transform(img)
    changed_img2.save(os.path.join(output_path, f"{mri_image}_changed_brightness_2.jpg"))
    #changed_img3 = brightness_transform(img)
    #changed_img3.save(os.path.join(output_path, f"{mri_image}_changed_brightness_3.jpg"))
    #changed_img4 = brightness_transform(img)
    #changed_img4.save(os.path.join(output_path, f"{mri_image}_changed_brightness_4.jpg"))


# %%
#getting the data ready using DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])
dataset = datasets.ImageFolder(root="Processed_MRI_Data_For_Second_Model", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataset.classes)

# %%
#Bring in pretrained resnet50
import torch
import torchvision.models as models
res_net_model = models.resnet50(pretrained=True)

# %%
#create the training set and validation set, but for each batch, weight the images so that
#we get close to an equal number of images of each class in each batch.
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.30, stratify=dataset.targets)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

#get the count of each class
class_counts = [0,0]
for class_i in train_indices:
    c_i = dataset.targets[class_i]
    class_counts[c_i] += 1
class_proportions = [1/class_counts[0] , 1/class_counts[1]]
class_proportions = [prop * 100000 for prop in class_proportions]
sample_weights = []
for idx in train_indices:  # Iterate through the training indices
    label = dataset.targets[idx]
    sample_weights.append(class_proportions[0] if label == 0 else class_proportions[1])
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=50, sampler=sampler)
#print(sample_weights)
eval_loader = DataLoader(val_dataset, batch_size = 50, shuffle = False)

num_batches_to_check = 10  # Number of batches you want to inspect


for i, (images, labels) in enumerate(train_loader):
    if i >= num_batches_to_check:
        break
    label_count = Counter(labels.tolist())  # Count the labels in the batch
    print(f"Batch {i + 1} label count:", label_count)
print("-----------------------------")
for i, (images, labels) in enumerate(eval_loader):
    if i >= num_batches_to_check:
        break
    label_count = Counter(labels.tolist())  # Count the labels in the batch
    print(f"Batch {i + 1} label count:", label_count)

# %%
#create output layer, get two output layers for this binary classification
import torch
import torchvision.models as models

res_net_model.fc = torch.nn.Linear(res_net_model.fc.in_features, 2)

# %%
#Define the loss function and the optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(res_net_model.parameters(), lr=0.000001)

# %%
'''
import torch
import torch.nn as nn
import torchvision.models as models
num_ftrs = res_net_model.fc.in_features
res_net_model.fc = nn.Sequential(
    nn.Dropout(p=0.4),  # 50% Dropout to prevent overfitting
    nn.Linear(num_ftrs, 2)  # Assuming 2 classes in your dataset
)

#begin forward and backwards propogation
from collections import Counter
num_epochs = 1
for epoch in range(num_epochs):
    res_net_model.train()
    running_loss = 0.0
    #iterate through a batch
    print(len(train_loader))
    for ind, (inputs, labels) in enumerate(train_loader):
        #print(ind)
        optimizer.zero_grad() #zero out gradients
        batch_outputs = res_net_model(inputs)
        loss = criterion(batch_outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        running_loss += loss.item()
        if (ind + 1) % 10 == 0:
            label_count = Counter(labels.tolist())  # Count the labels in the batch
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{ind+1}/{len(train_loader)}], Loss: {loss.item():.4f} label count: {label_count}")
        if(ind + 1) % 150 == 0:
            torch.save(res_net_model.state_dict(), f"Second_Model_checkpoint_step_{ind+1}.pth")
            print("Model weights saved")            

    #start evaluation
    res_net_model.eval()
    correct, total = 0 , 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            eval_outputs = res_net_model(inputs)
            _, predicted = torch.max(eval_outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
            
torch.save(res_net_model.state_dict(), "progress_model_weights.pth")
'''

# %%
#Main training algorithm I used
'''
#training algorithm I used while observing. Load in the weights previously gotten during the training. 
#train it, observe the loss each batch, every two batches, you can be asked to save the current weights.
#This allows us to view the loss closely, and I believe this allows us to reach the global minimum faster.
#add a drop out layer to prevent overfitting due to our unequal class sizes.
import torch
import torch.nn as nn
import torchvision.models as models
num_ftrs = res_net_model.fc.in_features
res_net_model.fc = nn.Sequential(
    nn.Dropout(p=0.4), 
    nn.Linear(num_ftrs, 2)  
)
res_net_model.load_state_dict(torch.load("Second_Model_checkpoint_step_150.pth"))
#begin forward and backwards propogation
from collections import Counter
num_epochs = 1
for epoch in range(num_epochs):
    res_net_model.train()
    running_loss = 0.0
    #iterate through a batch
    print(len(train_loader))
    for ind, (inputs, labels) in enumerate(train_loader):
        #print(ind)
        optimizer.zero_grad() #zero out gradients
        batch_outputs = res_net_model(inputs)
        loss = criterion(batch_outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        running_loss += loss.item()
        label_count = Counter(labels.tolist())
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{ind+1}/{len(train_loader)}], Loss: {loss.item():.4f} label count: {label_count}")
        if (ind + 1) % 10 == 0:
            save = input("Save this model?")
            if save == "y":
                torch.save(res_net_model.state_dict(), f"NEWLY_SAVED_MODEL_2_Updated_checkpoint_step_{ind+1}.pth")
                print("Model weights saved")                    

    #start evaluation
    res_net_model.eval()
    correct, total = 0 , 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            eval_outputs = res_net_model(inputs)
            _, predicted = torch.max(eval_outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
            
torch.save(res_net_model.state_dict(), "progress_model_weights.pth")
'''

# %%
#evaluate the model weights using the best weights we got
#add dropout layer to prevent overfitting
import torch
import torchvision.models as models
import torch.nn as nn
res_net_model = models.resnet50(pretrained = True)
res_net_model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(res_net_model.fc.in_features, 2)
)
res_net_model.load_state_dict(torch.load("NEWLY_SAVED_MODEL_2_Updated_checkpoint_step_40.pth"))
res_net_model.eval()
correct, total = 0 , 0
with torch.no_grad():
    for ind, (inputs, labels) in enumerate(eval_loader):
        eval_outputs = res_net_model(inputs)
        _, predicted = torch.max(eval_outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"Batch {ind+1}: Accuracy: {correct / total} , {correct}: {total}")

accuracy = 100 * correct / total
print(f"Evaluation Accuracy: {accuracy:.2f}%")


# %%



