#Prepare Images Script
#resize images and convert to RGB for DenseNet 121, 224 × 224
import os
#Create the output file directories
output_MRI_images = "Processed_MRI_Data"
sub_output_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
os.makedirs(output_MRI_images, exist_ok=True)

for sub_dir in sub_output_directories:
    new_path = os.path.join(output_MRI_images, sub_dir)
    os.makedirs(new_path, exist_ok= True)

# %%
#loop through input file directories, resize to 224x224 and convert from grayscale to RGB
from PIL import Image
input_MRI_images = "Alzhiehmer_MRI_Data"
sub_input_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
for ind,sub_dir in enumerate(sub_input_directories):
    dir_path = os.path.join(input_MRI_images, sub_dir)
    for mri_image in os.listdir(dir_path):
        if mri_image.lower().endswith((".jpg")):
            mri_image_path = os.path.join(dir_path, mri_image)
            image = Image.open(mri_image_path)
            image = image.convert("RGB")
            image = image.resize((224, 224))
            proper_output_folder = os.path.join(output_MRI_images , sub_output_directories[ind])
            new_file_name = os.path.join(proper_output_folder, mri_image)
            image.save(new_file_name)

        

# %%
#make sure we have enough images in the output directories
output_MRI_images = "Processed_MRI_Data"
sub_output_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
for sub_dir in sub_output_directories:
    new_path = os.path.join(output_MRI_images, sub_dir)
    print(len(os.listdir(new_path)))
print("-----------------------------")
input_MRI_images = "Alzhiehmer_MRI_Data"
sub_input_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
for sub_dir in sub_input_directories:
    new_path = os.path.join(input_MRI_images, sub_dir)
    print(len(os.listdir(new_path)))
    

# %%
#Now we will get the first model set up (Non Demented vs Any sign of Dementia)
import os
from PIL import Image
input_MRI_images = "Processed_MRI_Data"
sub_input_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]

output_MRI_images = "Processed_MRI_Data_For_First_Model"
sub_output_directories = ["Signs_of_Dementia" , "Non_Demented"]

for sub_dir in sub_output_directories:
    new_path = os.path.join(output_MRI_images, sub_dir)
    os.makedirs(new_path, exist_ok= True)


for ind,sub_dir in enumerate(sub_input_directories):
    dir_path = os.path.join(input_MRI_images, sub_dir)
    for mri_image in os.listdir(dir_path):
        if mri_image.lower().endswith((".jpg")):
            mri_image_path = os.path.join(dir_path, mri_image)
            image = Image.open(mri_image_path)
            if(ind != 3):
                proper_output_folder = os.path.join(output_MRI_images , sub_output_directories[0])
            else:
                proper_output_folder = os.path.join(output_MRI_images , sub_output_directories[1])
            new_file_name = os.path.join(proper_output_folder, mri_image)
            image.save(new_file_name)


# %%
#Sets up images for second model
import os
from PIL import Image
input_MRI_images = "Processed_MRI_Data"
sub_input_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
output_MRI_images = "Processed_MRI_Data_For_Second_Model"
sub_output_directories = ["Signs_of_Mild_Dementia" , "Signs_of_Moderate_Dimentia"]

for sub_dir in sub_output_directories:
    new_path = os.path.join(output_MRI_images, sub_dir)
    os.makedirs(new_path, exist_ok= True)

for ind,sub_dir in enumerate(sub_input_directories):
    if(ind == 3):
        continue
    dir_path = os.path.join(input_MRI_images, sub_dir)
    for mri_image in os.listdir(dir_path):
        if mri_image.lower().endswith((".jpg")):
            mri_image_path = os.path.join(dir_path, mri_image)
            image = Image.open(mri_image_path)
            if(ind != 1):
                proper_output_folder = os.path.join(output_MRI_images , sub_output_directories[0])
            else:
                proper_output_folder = os.path.join(output_MRI_images , sub_output_directories[1])
            new_file_name = os.path.join(proper_output_folder, mri_image)
            image.save(new_file_name)
            
            
            
        
   


# %%
#sets up images for third model
import os
from PIL import Image
input_MRI_images = "Processed_MRI_Data"
sub_input_directories = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia", "Non Demented"]
output_MRI_images = "Processed_MRI_Data_For_Third_Model"
sub_output_directories = ["Signs_of_Very_Mild_Dementia" , "Signs_of_Mild_Dimentia"]

for sub_dir in sub_output_directories:
    new_path = os.path.join(output_MRI_images, sub_dir)
    os.makedirs(new_path, exist_ok= True)

for ind,sub_dir in enumerate(sub_input_directories):
    if(ind == 3 or ind == 1):
        continue
    dir_path = os.path.join(input_MRI_images, sub_dir)
    for mri_image in os.listdir(dir_path):
        if mri_image.lower().endswith((".jpg")):
            mri_image_path = os.path.join(dir_path, mri_image)
            image = Image.open(mri_image_path)
            if(ind != 0):
                proper_output_folder = os.path.join(output_MRI_images , sub_output_directories[0])
            else:
                proper_output_folder = os.path.join(output_MRI_images , sub_output_directories[1])
            new_file_name = os.path.join(proper_output_folder, mri_image)
            image.save(new_file_name)

# %%



