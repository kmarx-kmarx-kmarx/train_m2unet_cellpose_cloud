import os
from interactive_m2unet import M2UnetInteractiveModel
import numpy as np
import albumentations as A  
from tqdm import trange
import gcsfs
from utils import *
from cellpose import models, utils
import json
import random
from skimage.filters import threshold_otsu
import cv2
import csv
# specify the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

def main():
    random.seed(1)
    # modify these:
    # cellpose model
    cellpose_model_path = 'cp_dpc_new'
    # gcs data source
    gcs_project = 'soe-octopi'
    gcs_token = '/home/prakashlab/Documents/kmarx/data-20220317-keys.json'  
    bucket_source = "gs://octopi-malaria-tanzania-2021-data/Negative-Donor-Samples"#'gs://octopi-malaria-uganda-2022-data/'
    local_path = 'data_tanzania'
    dataset_file = 'list of datasets.txt'
    save_interval = 50
    n_views_train = 101 # number of views from each dataset to train on
    n_views_valid = 50  # number of views from each dataset to run validation on
                        # make (n_views_valid + n_views_train) a prime number for good cross-validation
    random_views = True # randomize the training data
    erode_mask = 1      # erode mask to improve cell separation
    # illumination correction
    flatfield_left = np.load('flatfield_left.npy')
    flatfield_right = np.load('flatfield_right.npy')
    # m2unet training
    epochs = 2000
    sz = 1024
    model_root = "m2unet_model_flat_less_erode"
    transform = A.Compose(
        [
            A.RandomCrop(1500, 1500),
            A.Rotate(limit=(-10, 10), p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CenterCrop(sz, sz),  
        ]
    )
    model_config = {
        "type": "m2unet",
        "activation": "sigmoid",
        "output_channels": 1,
        "loss": {"name": "BCELoss", "kwargs": {}},
        "optimizer": {"name": "RMSprop", "kwargs": {"lr": 1e-2, "weight_decay": 1e-8, "momentum": 0.9}},
        "augmentation": A.to_dict(transform),
    }

    # prep for ML
    torch.cuda.empty_cache()

    # connect to google cloud
    with open(dataset_file,'r') as f:
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')[0:-1]
    fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)
    indices = {}
    for dataset in DATASET_ID:
        # Get the total number of views and create an index dict
        print(bucket_source + '/' + dataset + '/acquisition parameters.json')
        json_file = fs.cat(bucket_source + '/' + dataset + '/acquisition parameters.json')
        acquisition_parameters = json.loads(json_file)
        n_images = acquisition_parameters['Ny'] * acquisition_parameters['Nx']
        i = list(range(n_images))
        if random_views:
            random.shuffle(i)
        indices[dataset] = i
        indices[dataset+'Ny'] = acquisition_parameters['Ny']
        indices[dataset+'Nx'] = acquisition_parameters['Nx']
    # We won't see all the views
    imax = min([len(indices[dataset]) for dataset in DATASET_ID])
    # initialize cellpose
    model_cp = models.CellposeModel(gpu=True, pretrained_model=cellpose_model_path)

    # initialize m2unet
    model_un = M2UnetInteractiveModel(
        model_config=model_config,
        model_dir=model_root,
        resume=False,
        pretrained_model=None,
        default_save_path=os.path.join(model_root, "model.pth"),
    )

    # start training
    i = 0
    min_loss = np.Inf
    max_loss = -np.Inf
    threshold_stack = []
    for e in trange(epochs):
        # losses = np.zeros(n_views_train * len(DATASET_ID))
        for l in trange(n_views_train):
            i += 1
            i %= imax
            for j, dataset in enumerate(DATASET_ID):
                # get target image
                im, mask = get_im_mask(i, indices, dataset, fs, bucket_source, flatfield_left, flatfield_right, model_cp, local=local_path)
                mask = mask/np.max(mask)
                if erode_mask > 0:
                    shape = cv2.MORPH_ELLIPSE
                    element = cv2.getStructuringElement(shape, (2 * erode_mask + 1, 2 * erode_mask + 1), (erode_mask, erode_mask))
                    mask = np.array(cv2.erode(mask, element))
                im = (im - np.mean(im)) /np.std(im)
                im = np.stack([im,]*1, axis=2)
                labels = model_un.transform_labels(mask)
                labels = np.expand_dims(labels, axis=0)
                labels = np.expand_dims(labels, axis=-1)
                im = np.expand_dims(im, axis=0)
                k = len(DATASET_ID)*l + j
                model_un.train_on_batch(im, labels)
        losses = np.zeros(n_views_valid * len(DATASET_ID))
        for l in trange(n_views_valid):
            i += 1
            i %= imax
            for j, dataset in enumerate(DATASET_ID):
                # get target image
                im, mask = get_im_mask(i, indices, dataset, fs, bucket_source, flatfield_left, flatfield_right, model_cp, local=local_path)
                im = im[:sz, :sz]
                mask = mask[:sz, :sz]
                mask = mask/np.max(mask)
                if erode_mask > 0:
                    shape = cv2.MORPH_ELLIPSE
                    element = cv2.getStructuringElement(shape, (2 * erode_mask + 1, 2 * erode_mask + 1), (erode_mask, erode_mask))
                    mask = np.array(cv2.erode(mask, element))
                im = (im - np.mean(im)) /np.std(im)
                im = np.stack([im,]*1, axis=2)
                im = np.expand_dims(im, axis=0)
                mask =  np.stack([mask,], axis=2)
                mask = np.expand_dims(mask, axis=0)
                loss, result = model_un.get_loss(im, mask)
                threshold_stack.append(threshold_otsu(255*result[0]))
                k = len(DATASET_ID)*l + j
                losses[k] = loss
        
        # Look at average loss - if it's higher than before, save it
        avg = np.mean(losses) * 100
        print(f"\n\n{e}: {avg}\n")
        if avg < min_loss:
            min_loss = avg
            model_un.save(file_path=os.path.join(model_root, f"model_{e}_{int(min_loss)}.pth"))
        elif avg > max_loss:
            max_loss = avg
            model_un.save(file_path=os.path.join(model_root, f"model_{e}_{int(max_loss)}.pth"))
        elif e % save_interval == 0:
            model_un.save(file_path=os.path.join(model_root, f"model_{e}_{int(avg)}.pth"))

    # save at end
    model_un.save(file_path=os.path.join(model_root, f"model_{e}_{int(avg)}.pth"))
    threshold_stack = np.array(threshold_stack)
    threshold_stack.tofile(os.path.join(model_root,'thresholds.csv'), sep = '\n')

if __name__ == "__main__":
    main()