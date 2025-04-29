
#Third Model: Mild Dementia vs Signs of very mild Dementia
#getting the data for the first model ready using DataLoader
#add a few images for moderate but changing brightness 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])
dataset = datasets.ImageFolder(root="Processed_MRI_Data_For_Third_Model", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataset.classes)

# %%
#Bring in Resnet pretrained model
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

train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.25, stratify=dataset.targets)
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
#add output layer for the two possible outputs
import torch
import torchvision.models as models

res_net_model.fc = torch.nn.Linear(res_net_model.fc.in_features, 2)

# %%
#Define the loss function and the optimizer
#same learning rate strategy as the first two models
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
        if((ind + 1) % 150 == 0 or (ind+1) % len(train_loader)==0):
            torch.save(res_net_model.state_dict(), f"Third_Model_checkpoint_step_{ind+1}.pth")
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
#main training algorithm I used 
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
    nn.Dropout(p=0.4),  # 50% Dropout to prevent overfitting
    nn.Linear(num_ftrs, 2)  # Assuming 2 classes in your dataset
)
res_net_model.load_state_dict(torch.load("Third_Model_checkpoint_step_281.pth"))
from collections import Counter
num_epochs = 1
for epoch in range(num_epochs):
    res_net_model.train()
    running_loss = 0.0
    #iterate through a batch
    print(len(train_loader))
    for ind, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad() #zero out gradients
        batch_outputs = res_net_model(inputs)
        loss = criterion(batch_outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        running_loss += loss.item()
        label_count = Counter(labels.tolist())  # Count the labels in the batch
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{ind+1}/{len(train_loader)}], Loss: {loss.item():.5f} label count: {label_count}")
        if (ind + 1) % 20 == 0:
            save = input("Save this model?")
            if save == "y":
                torch.save(res_net_model.state_dict(), f"NEWLY_SAVED_MODEL_3_Updated_checkpoint_step_{ind+1}.pth")
                print("Model weights saved")            
            

            
torch.save(res_net_model.state_dict(), "progress_model_weights.pth")
'''

# %%
#evaluate the model weights with our best model weights found
import torch
import torchvision.models as models
import torch.nn as nn
res_net_model = models.resnet50(pretrained = True)
res_net_model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(res_net_model.fc.in_features, 2)
)
res_net_model.load_state_dict(torch.load("NEWLY_SAVED_MODEL_3_Updated_checkpoint_step_280.pth"))
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



