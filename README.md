# Dementia Prediction from MRI Images using OASIS Dataset
## Overview

This project provides a system for predicting dementia from MRI images using a combination of convolutional neural networks (CNN) and random forest classifiers. The system processes MRI images to predict dementia, ensuring a non-discriminatory and privacy-conscious approach. It utilizes a CNN to extract dementia probabilities for each MRI image type, and these probabilities are then combined and enhanced using a random forest model to provide robust and accurate predictions.
 <!-- Add a relevant image in the `docs` folder -->
## Key Features

* A set of Convolution Neural Networks process MRI images to extract dementia probabilities for different image types.
* Combines the CNN outputs to form a comprehensive prediction model using a random forest classifier.
* Ensures privacy by optionally not requiring personal demographic data.
* Does not require high computational resources, able to be run on laptops or on Colab.

## Origin of Data
This project uses data from ([OASIS-1 DATASET](https://sites.wustl.edu/oasisbrains/)) aimed at making neuroimaging datasets freely available to the scientific community. This set consists of a cross-sectional collection of 416 subjects aged 18 to 96. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. 100 of the included subjects over the age of 60 have been clinically diagnosed with very mild to moderate Alzheimer’s disease (AD). 
For easier access 
