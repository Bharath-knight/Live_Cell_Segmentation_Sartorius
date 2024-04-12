# Live_Cell_Segmentation_Sartorius
In this study, we explore the performance of three deep learning models, UNet, Mask R-CNN, and CellSAM for the LiveCELL dataset which is part of Sartorius - Cell Instance Segmentation Kaggle challenge using Pytorch implementation. 

# Introduction:
The LiveCELL dataset used in this work is a large collection of over 150,000 microscopy images containing neuronal cells. The LiveCELL dataset is part of the Sartorius - Cell Instance Segmentation Kaggle Competition hosted by a life science company, Sartorius. This competition mainly focuses on cell instance segmentation within microscopy images. This Challenge conducted in 2021 aims to advance cell segmentation techniques in biomedical research. 
LIVECell is the predecessor dataset to this competition. It is a high-quality, manually annotated, and expert-validated dataset with 1.6 million annotations of 8 different cell types.<br/><br/> LIVECell dataset is provided in JSON format but it is converted to a CSV file for convenience. 
There are 5239 images and 1662447 annotations in the LIVECell dataset. The average annotations per image is 317.32 which is much higher than the average annotations per image in the competition dataset since the cell cultures are more confluent in LIVECell dataset.
As the LIVECell dataset is the predecessor of the competition dataset, they are very similar. Images have a width of 704 and a height of 520 pixels in both datasets and they are probably taken from the same source

# Data Preprocessing:  

The images were of the dimensions 704 X 520, and I  normalized them as it's important to standardize pixel values in images. Pre-calculated mean and standard deviation values with ResNet-specific normalization were applied where mean pixel values were subtracted from each channel and each channel was divided by its standard deviation. <br/>

I utilized Run-Length encoded (RLE) compression for storage efficiency in dealing with masks where I compressed binary masks by storing consecutive sequences of identical values. I further converted the RLE mask string to NumPy arrays to load into the model.<br/><br/>

The augmentation steps that I used were, Normalizing the images, and flipping the images horizontally and vertically, using functions VerticalFlip and HorizontalFlip for now. <br/><br/>

# Models:

## U-Net:

Optimizer: The optimizer used was the Adam optimizer. 
Learning rate: The learning rate was set to 0.0005 or 5e-4 (A lower learning rate will result in a more gradual update of the parameters)
Batch size: The batch size is the number of images that are processed at a time during training. I tried to use a larger batch size of “64” 
Number of epochs: The number of epochs is the number of times that the model is trained on the entire training dataset. I tried going with a higher number of epochs (30)  which resulted in overfitting, so I went with 12 epochs. 

Results: 
Average Precision (AP) @[ IoU=0.50} = 0.644 with the maximum number of detections per image considered to be 2000 
Average Recall (AR) @[ IoU=0.50:0.95 ] = 0.490 with the maximum number of detections per image considered to 2000 
Training for 12 epochs, Train Loss of 1.5086 was obtained
Time of computation - 643.9s

## Mask R-CNN:

Optimizer: The optimizer used was SGD (Stochastic Gradient Descent). 
Learning rate: The learning rate was set to 0.001 
Batch size: The batch size is the number of images that are processed at a time during training. I tried to use a larger batch size “2” as it was less computationally expensive 
Number of epochs: The number of epochs is the number of times that the model is trained on the entire training dataset. I tried going with a higher number of epochs (30)  which overfitted, so I went with 8 epochs. 

Results: 
Average Precision (AP) @[ IoU=0.50} = 0.764 with the maximum number of detections per image considered to be 2000 
Average Recall (AR) @[ IoU=0.50] = 0.57 with the maximum number of detections per image considered to 2000 
Training for 12 epochs, Train Loss of 1.250 was obtained
Time of computation - 2139.4s  

Kaggle Competition Submission:
Earned an evaluation score of 0.277 in the competition while the best score in the competition was 0.356

## CellSAM model:

Results:
Average Precision (AP) @[ IoU=0.50} = 0.594 with the maximum number of detections per image considered to be 2000 
Average Recall (AR) @[ IoU=0.50:0.95 ] = 0.38 with the maximum number of detections per image considered to 2000 
For Individual Bounding Box Segmentations:
Average Predicted IOU:  0.9344914555549622
Average Stability score:  0.9438048005104065

