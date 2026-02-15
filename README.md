# AutoEncoder-SR

This project is my first attempt to build an autoencoder-based image super-resolution model.  
The goal is to understand how reconstruction networks learn image representations and recover higher-resolution details from low-resolution inputs.

## Project Status
This is an experimental implementation for learning and testing ideas.

## Model Overview
The model follows a standard autoencoder pipeline:

Low-Resolution Image > Encoder > Latent Representation > Decoder > Super-Resolved Image

The encoder compresses the input image into a compact feature space,  
and the decoder reconstructs a higher-quality version from these features.

## Architecture
- Convolutional layers for feature extraction  
- Latent bottleneck representation  
- Decoder for image reconstruction  

(Details can be found in the notebook.)

## Implementation
The full training and testing pipeline is implemented in: model_test.ipynb

