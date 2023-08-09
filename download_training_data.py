import os
import numpy as np
from tqdm import tqdm
from itertools import product
import gcsfs
from utils import *
from cellpose import models
import json
import random

def main():
    # modify these:
    # gcs data source
    gcs_project = 'soe-octopi'
    gcs_token = '/home/prakashlab/Documents/kmarx/data-20220317-keys.json'  
    bucket_source = 'gs://octopi-sbc-segmentation'#"gs://octopi-malaria-tanzania-2021-data"#
    local_path = 'data_sbc_validation'
    dataset_file = 'validation dataset.txt'
    # illumination correction
    flatfield_left = np.load('flatfield_left.npy')
    flatfield_right = np.load('flatfield_right.npy')
    # select only middle of smear - ignore the front and end of smear that give bad results
    mid_smear = False
    randomize = False
    save_remote = True
    n_im = 0 # images per dataset, set to 0 for all
    # cellpose
    cellpose_model_path = 'cp_dpc_new'

    os.makedirs(local_path, exist_ok=True)

    # connect to google cloud
    with open(dataset_file,'r') as f:
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')[0:-1]
    fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)
    # init cellpose
    model_cp = models.CellposeModel(gpu=True, pretrained_model=cellpose_model_path)
    for dataset in tqdm(DATASET_ID):
        # Get the total number of views and create an index dict
        print(bucket_source + '/' + dataset + '/acquisition parameters.json')
        json_file = fs.cat(bucket_source + '/' + dataset + '/acquisition parameters.json')
        acquisition_parameters = json.loads(json_file)

        # Pick which range of x, y to use
        # We want the first ID 
        xrng = None
        yrng = None
        if mid_smear:
            xmin = acquisition_parameters['Ny']//3
            xmax = xmin * 2
            xrng = range(xmin, xmax)
            yrng = range(acquisition_parameters['Nx'])
        else:
            xrng = range(acquisition_parameters['Ny'])
            yrng = range(acquisition_parameters['Nx'])
        krng = range(1)
        
        indices = list(product(xrng, yrng, krng))
        if randomize:
            random.shuffle(indices)
        if n_im > 0:
            indices = indices[0:n_im]

        for x, y, k in tqdm(indices):
            file_id = str(x) + '_' + str(y) + '_' + str(k)
            try:
                load_from_gcs(fs, bucket_source, dataset, file_id, flatfield_left, flatfield_right, model_cp, local=local_path, save_remote=save_remote) 
            except FileNotFoundError:
                print(f"File {file_id} from {dataset} not found")

if __name__ == "__main__":
    main()