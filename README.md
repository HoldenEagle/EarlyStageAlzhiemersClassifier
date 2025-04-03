The code in this repo creates a multi model deep learning ai mechanism that correctly analyzes and detects early signs of Alzheimer's disease with over 98% accuracy. 
This ai mechanism was tested using MRI images from the OASIS MRI Dataset that contains over 80,000 brain structure MRI images. These brain images were sliced along the z-axis into 256 peices,
and slices ranging from 100 to 160 were included in this dataset. There are four classes that each of these images could be classified into, Moderate Demented/Demented , Mild demented, 
Very Mild Demented, and Non-Demented. The dataset claims these classifications are based on the Clinical Dementia Rating (CDR) values.

To achieve the performance that I did, I created three neural networks, each using fine tuning from pretrained ResNet50 networks. These networks each did a binary classification.
For example, the first model classified between non dementia and any sign of dementia. The second model classified between moderate dimentia and signs of either mild or very mild
dementia, and the third model classified between mild dementia and very mild dementia. Together these models can work in a sequential way, and we can take an image and run it through 
these models in this order to find what classification it should recieve. All three of these models can predict these binary classifications with over 99 percent accuracy, and 
together the accuracy is around 97-98 percent. These models were trained individually and only needed a few epochs to achieve this level of accuracy, showing that multi models like this
can not only drastically improve performance, but can efficiently and quickly achieve this level of performance. This method could be expanded to other problems in Radiology, and may be able to 
help accurately classify other issues that can be found structurally in the brain.

How to Run this Mechanism:
1. Clone the Repository
2. open up Jupyter-Lab
3. run PrepareImages.ipynb to see how the binary classification data is created from the original Alzhiermers dataset
4. The three models are included in FirstModel, SecondModel, and ThirdModel. The weights of the most optimal networks for each that I could find are included in this repo. However, you can see how I
   trained these models, and the last cells show the accuracy of each network when evaluated on the test set for their specific binary classification.
5. FullModel.ipynb shows all three models working together. You can run through that to see how these models all together work when classifying the data. The final cell shows this process, and tries to get
   2000 or as many images from each class it can get to test. However, you can change this number and you can even take this section out so you can see the full model's results on all of the dataset.
   
