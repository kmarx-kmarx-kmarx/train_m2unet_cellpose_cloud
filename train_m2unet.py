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
import cv2
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
    bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
    local_path = 'data'
    dataset_file = 'list of datasets.txt'
    save_interval = 10
    n_views_train = 101 # number of views from each dataset to train on
    n_views_valid = 50  # number of views from each dataset to run validation on
                        # make (n_views_valid + n_views_train) a prime number for good cross-validation
    random_views = True # randomize the training data
    erode_mask = 0      # erode mask to improve cell separation
    # illumination correction
    flatfield_left = np.load('flatfield_left.npy')
    flatfield_right = np.load('flatfield_right.npy')
    # m2unet training
    epochs = 1000
    sz = 1024
    model_root = "m2unet_model_7"
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
        json_file = fs.cat(bucket_source + '/' + dataset + '/acquisition parameters.json')
        acquisition_parameters = json.loads(json_file)
        n_images = acquisition_parameters['Ny'] * acquisition_parameters['Nx']
        i = list(range(n_images))
        if random_views:
            random.shuffle(i)
        indices[dataset] = i
        indices[dataset+'Ny'] = acquisition_parameters['Ny']
        indices[dataset+'Nx'] = acquisition_parameters['Nx']
    imax = max([len(indices[dataset]) for dataset in DATASET_ID])
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
                im = np.stack([im,]*3, axis=2)
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
                im = np.stack([im,]*3, axis=2)
                im = np.expand_dims(im, axis=0)
                mask =  np.stack([mask,], axis=2)
                mask = np.expand_dims(mask, axis=0)
                loss, __ = model_un.get_loss(im, mask)
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

def get_im_mask(i, indices, dataset, fs, bucket_source, flatfield_left, flatfield_right, model_cp, local=None):
    idx = indices[dataset][i]
    x = int(idx/indices[dataset+'Nx'])
    y = idx % indices[dataset+'Ny']
    k = 0
    # check if file exists
    file_id = str(x) + '_' + str(y) + '_' + str(k)
    filepath = os.path.join(local, f"{dataset}_{file_id}_seg.npz")
    if os.path.exists(filepath):
        im, mask = load_from_file(filepath)
    else:
        os.makedirs(local, exist_ok=True)
        im, mask = load_from_gcs(fs, bucket_source, dataset, file_id, flatfield_left, flatfield_right, model_cp, local=local)  
    
    return im, mask

def load_from_file(filepath):
    items = np.load(filepath, allow_pickle=True)
    return items['img'], items['mask']

def load_from_gcs(fs, bucket_source, dataset, file_id, flatfield_left, flatfield_right, model_cp, local=None):
    # generate DPC
    I_BF_left = imread_gcsfs(fs,bucket_source + '/' + dataset + '/0/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp')
    I_BF_right = imread_gcsfs(fs,bucket_source + '/' + dataset + '/0/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp')
    if len(I_BF_left.shape)==3: # convert to mono if color
        I_BF_left = I_BF_left[:,:,1]
        I_BF_right = I_BF_right[:,:,1]
    I_BF_left = I_BF_left.astype('float')/255
    I_BF_right = I_BF_right.astype('float')/255
    # flatfield correction
    I_BF_left = I_BF_left/flatfield_left
    I_BF_right = I_BF_right/flatfield_right
    I_DPC = generate_dpc(I_BF_left,I_BF_right)

    # cellpose preprocessing
    im = I_DPC - np.min(I_DPC)
    im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
    # run segmentation
    mask, flows, styles = model_cp.eval(im, diameter=None)

    outlines = mask * utils.masks_to_outlines(mask)
    mask = (mask  > 0) * 1.0
    outlines = (outlines  > 0) * 1.0
    mask = (mask * (1.0 - outlines) * 255).astype(np.uint8)

    I_DPC = (255*I_DPC).astype(np.uint8)

    if local != None:
        savepath = os.path.join(local, f"{dataset}_{file_id}_seg.npz")
        # io.masks_flows_to_seg(I_DPC, mask, flows, 0, savepath, [0,0])
        np.savez(savepath, mask=mask, img=I_DPC)

    return I_DPC, mask      

if __name__ == "__main__":
    main()