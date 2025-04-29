#Full_Model_Script
#Use the transform class for each image
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])



# %%
#loading all the binary classifiers from each model , mild vs very mild, mild vs moderate, 
#non demented vs signs of dementia. Using the weights we got from training. 
import torch
import torchvision.models as models
import torch.nn as nn
very_mild_vs_mild = models.resnet50(pretrained=True)
very_mild_vs_mild.fc = nn.Sequential(
    nn.Dropout(p=0.4), 
    nn.Linear(very_mild_vs_mild.fc.in_features, 2)  
)
very_mild_vs_mild.load_state_dict(torch.load("NEWLY_SAVED_MODEL_3_Updated_checkpoint_step_280.pth"))
very_mild_vs_mild.eval()

mild_vs_moderate = models.resnet50(pretrained = True)
mild_vs_moderate.fc = nn.Sequential(
    nn.Dropout(p=0.4), 
    nn.Linear(mild_vs_moderate.fc.in_features, 2)  
)
mild_vs_moderate.load_state_dict(torch.load("NEWLY_SAVED_MODEL_2_Updated_checkpoint_step_40.pth"))
mild_vs_moderate.eval()

non_demented_vs_dementia = models.resnet50(pretrained = True)
non_demented_vs_dementia.fc = torch.nn.Linear(non_demented_vs_dementia.fc.in_features,2)
non_demented_vs_dementia.load_state_dict(torch.load("NEWLY_SAVED_MODEL_1_Part_2_Updated_checkpoint_step_500.pth"))
non_demented_vs_dementia.eval()


# %%
#Predict function, takes in a image and the model, and we make a prediction. We then take the 
#maximum confident class as the answer.
def predict(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        logits = model(image)
    predicted_class = torch.argmax(logits, dim=1).item()
    #print(predicted_class)
    return predicted_class

# %%
#MAIN CODE OF OUR MULTI MODEL MECHANISM. For this, we are testing 2000 or as many images of we can from each of the four 
#classes. It first checks if the image shows signs of any dementia using our first model. Then, if it does, we check using
#our next model to see if the image shows signs of moderate dementia, if not, we go to our third and final model, which
#checks if the model shows signs of very mild dementia or mild dementia. We then can compare this to the correct label, 
#this code checks the accuracy of this approach, and we can check the accuracy of how many images we want by adjusting the 
#number from 2000 to anything.
import os
#Create the output file directories
input_MRI_images = "Processed_MRI_Data"
sub_input_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
#sub_input_directories = ["Very mild Dementia"]
num_correct = 0
total = 0
for ind,sub_dir in enumerate(sub_input_directories):
    sub_correct = 0
    sub_total = 0
    print(sub_dir)
    print("----------")
    index = 0
    dir_path = os.path.join(input_MRI_images, sub_dir)
    for mri_image in os.listdir(dir_path):
        if(index > 2000):
            continue
        if mri_image.lower().endswith((".jpg")):
            mri_image_path = os.path.join(dir_path, mri_image)
            pred = predict(mri_image_path, non_demented_vs_dementia)
            if(pred == 0):
                pred = "Non Demented"
            else:
                pred = predict(mri_image_path, mild_vs_moderate)
                if(pred == 1):
                    pred = "Moderate Dementia"
                else:
                    pred = predict(mri_image_path, very_mild_vs_mild)
                    if(pred== 1):
                        pred = "Very mild Dementia"
                    else:
                        pred = "Mild Dementia"
        #if(total % 20 == 0):
            #print(pred, sub_dir)
        if(pred == sub_dir):
             num_correct += 1
             sub_correct += 1
        total += 1
        sub_total += 1
        if(total % 100 == 0):
            #print(f"Accuracy Check: {num_correct/ total}")
            print(f"Accuracy for {sub_dir} : {sub_correct/sub_total}")
        index += 1
    print(f"Accuracy for {sub_dir} : {sub_correct/sub_total}")
print(f"Total Accuracy: {num_correct/total}")



# %%
'''
import os
#Create the output file directories
input_MRI_images = "Processed_MRI_Data"
sub_input_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
#sub_input_directories = ["Moderate Dementia"]
num_correct = 0
total = 0
for ind,sub_dir in enumerate(sub_input_directories):
    print(sub_dir)
    print("----------")
    index = 0
    dir_path = os.path.join(input_MRI_images, sub_dir)
    for mri_image in os.listdir(dir_path):
        if(index > 2000):
            continue
        votes = [0,0,0,0]
        votes_pred = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
        if mri_image.lower().endswith((".jpg")):
            mri_image_path = os.path.join(dir_path, mri_image)
            for v in range(3):
                pred = predict(mri_image_path, non_demented_vs_dementia)
                if(pred == 0):
                    votes[3] += 1
                else:
                    pred = predict(mri_image_path, mild_vs_moderate)
                    if(pred == 1):
                        votes[1] += 1
                    else:
                        pred = predict(mri_image_path, very_mild_vs_mild)
                        if(pred== 1):
                            votes[2] += 1
                        else:
                            votes[0] += 1
        max_vote, max_vote_index = 0,0
        for i, vote in enumerate(votes):
            if(vote > max_vote):
                max_vote = vote
                max_vote_index = i
        pred = votes_pred[max_vote_index]
        if(total % 20 == 0):
            print(pred, sub_dir , votes)
        if(pred == sub_dir):
             num_correct += 1
        total += 1
        if(total % 20 == 0):
            print(f"Accuracy Check: {num_correct/ total}")
        index += 1
print(f"Total Accuracy: {num_correct/total}")
'''

# %%



