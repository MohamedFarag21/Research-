# UNet Epistemic Uncertainty Quantification with Monte Carlo Dropout

This Python script is used to implement a UNet model for semantic segmentation tasks, with the addition of Monte Carlo Dropout for epistemic uncertainty quantification. The script is divided into several sections, each of which is explained below.

## Importing Libraries

The script begins by importing the necessary libraries and modules. These include PyTorch, torchvision, PIL, pandas, os, tifffile, cv2, tqdm, and numpy. The random seed is also set to 78 for reproducibility.

## Data Settings

The paths to the CSV file containing the image file names, the folder containing the input images, and the folder containing the target segmentation maps are defined. The ratio of the total dataset to be used for training and the batch size are also set.

## Transformations

A function is defined to create a transformation pipeline for the input images. The pipeline consists of a single operation, which is converting the input image to a PyTorch tensor. The same transformation is applied to the target images.

## Custom Dataset

A custom PyTorch Dataset class is defined to handle the loading and preprocessing of the images and target segmentation maps. The class takes the paths to the CSV file, image folder, and target folder as inputs, as well as optional transformations to be applied to the images and targets.

## Model Settings

The UNet model is defined with a specific architecture. The model takes as input the number of input channels, the number of output classes, and a list of the number of channels for each layer. The model includes both convolutional and deconvolutional blocks, as well as max pooling, upsampling, and dropout layers.

## Training Settings

A function is defined to set up the model for training. The model is moved to the specified device (default is CUDA), the optimizer is initialized with the specified learning rate (default is Adam with a learning rate of 0.001), and the loss function is set up (default is cross entropy loss).

## Monte Carlo Dropout

A function is defined to perform Monte Carlo Dropout for a single batch of data. The function generates multiple predictions for each sample in the batch by running the model with dropout enabled during inference. The predictions are then stored for uncertainty quantification.

## Calculating the Standard Deviation

The standard deviation of the Monte Carlo Dropout predictions is calculated for each class. The standard deviation values are then scaled to the range of -1 to 1 and the mean standard deviation for each class across all images is computed. The results are stored in a dictionary where the key is the class index, and the values are a list with the first element being the class name and the second element being the mean standard deviation value.