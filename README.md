# Train M2UNet Image Segmentation Model with Cellpose Outputs

This script downloads images from Google Cloud Storage, preprocesses them, and runs a pretrained Cellpose model to segment it. 
Cellpose is accurate and slow so here we are using it to train an M2UNet segmentation model which is faster but less accurate.

We have a large dataset containing Red Blood Cell (RBC) images. To make full use of the dataset, each epoch we are training and validating on a different set of images. 

We first initialize the datasets, given in a `list_of_datasets.txt` file in the same directory as the python script. For each dataset, we get the number of images in the dataset and create a list of indices to index through the datasets. We then randomize the order of the indices to uniformly sample the dataset without repeats.

Each epoch, we read the next `n_views_train` images from each dataset and run training on it. We next validate the training by reading the next `n_views_valid` images and finding the average loss. If the loss is less than the previous loss then we save the model before going to the next epoch.

Processed images are stored locally in `.npz` format along with the Cellpose-segmented masks.
