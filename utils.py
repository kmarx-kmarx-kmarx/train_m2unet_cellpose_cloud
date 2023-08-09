import imageio
import cv2
# import cupy as cp # conda install -c conda-forge cupy==10.2
# import cupyx.scipy.ndimage
import numpy as np
from scipy import signal
import pandas as pd
import xarray as xr
import gcsfs
from cellpose import utils, io
import os

def imread_gcsfs(fs,file_path):
    img_bytes = fs.cat(file_path)
    I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
    return I

def generate_dpc(I1,I2,use_gpu=False):
    if use_gpu:
        # img_dpc = cp.divide(img_left_gpu - img_right_gpu, img_left_gpu + img_right_gpu)
        # to add
        I_dpc = 0
    else:
        I_dpc = np.divide(I1-I2,I1+I2)
        I_dpc = I_dpc + 0.5
    I_dpc[I_dpc<0] = 0
    I_dpc[I_dpc>1] = 1
    return I_dpc

def get_im_mask(i, indices, dataset, fs, bucket_source, flatfield_left, flatfield_right, model_cp, only_middle_third=False, local=None):
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

def load_from_gcs(fs, bucket_source, dataset, file_id, flatfield_left, flatfield_right, model_cp, local=None, save_remote=False):
    # generate DPC
    bp = os.path.join(bucket_source, dataset, '0')
    I_BF_left = imread_gcsfs(fs, os.path.join(bp, file_id + '_' + 'BF_LED_matrix_left_half.bmp'))
    I_BF_right = imread_gcsfs(fs,os.path.join(bp, file_id + '_' + 'BF_LED_matrix_right_half.bmp'))
    if len(I_BF_left.shape)==3: # convert to mono if color
        I_BF_left = I_BF_left[:,:,1]
        I_BF_right = I_BF_right[:,:,1]
    I_BF_left = I_BF_left.astype('float')/255
    I_BF_right = I_BF_right.astype('float')/255
    # flatfield correction
    I_BF_left = I_BF_left/flatfield_left
    I_BF_right = I_BF_right/flatfield_right
    I_DPC = generate_dpc(I_BF_left,I_BF_right)
    if save_remote:
        # Save the DPC to remote
        save_remote_name = os.path.join(bp, file_id + '_' + 'DPC.bmp')
        with fs.open(save_remote_name, "wb") as f:
            f.write(cv2.imencode('.bmp',(255*I_DPC).astype(np.uint8))[1].tobytes())


    # cellpose preprocessing
    im = I_DPC - np.min(I_DPC)
    im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))

    # run segmentation
    mask, flows, styles = model_cp.eval(im, diameter=None)
    if save_remote:
        io.masks_flows_to_seg(im, mask, flows, 0, "./placeholder.bmp", [0,0])
        save_remote_name = os.path.join(bp, file_id + '_' + 'seg.npy')
        fs.put("./placeholder_seg.npy", save_remote_name)
        os.remove("./placeholder_seg.npy")

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

def export_spot_images_from_fov(I_fluorescence,I_dpc,spot_data,parameters,settings,gcs_settings,dir_out=None,r=30,generate_separate_images=False):
    pass
    # make I_dpc RGB
    if(len(I_dpc.shape)==3):
        # I_dpc_RGB = I_dpc
        I_dpc = I_dpc[:,:,1]
    else:
        # I_dpc_RGB = np.dstack((I_dpc,I_dpc,I_dpc))
        pass
    # get overlay
    # I_overlay = 0.64*I_fluorescence + 0.36*I_dpc_RGB
    # get the full image size
    height,width,channels = I_fluorescence.shape
    # go through spot
    counter = 0
    
    for idx, entry in spot_data.iterrows():
        # get coordinate
        i = int(entry['FOV_row'])
        j = int(entry['FOV_col'])
        x = int(entry['x'])
        y = int(entry['y'])
        # create the arrays for cropped images
        I_DPC_cropped = np.zeros((2*r+1,2*r+1), np.float)
        I_fluorescence_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
        # I_overlay_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
        # identify cropping region in the full FOV 
        x_start = max(0,x-r)
        x_end = min(x+r,width-1)
        y_start = max(0,y-r)
        y_end = min(y+r,height-1)
        x_idx_FOV = slice(x_start,x_end+1)
        y_idx_FOV = slice(y_start,y_end+1)
        # identify cropping region in the cropped images
        x_cropped_start = x_start - (x-r)
        x_cropped_end = (2*r+1-1) - ((x+r)-x_end)
        y_cropped_start = y_start - (y-r)
        y_cropped_end = (2*r+1-1) - ((y+r)-y_end)
        x_idx_cropped = slice(x_cropped_start,x_cropped_end+1)
        y_idx_cropped = slice(y_cropped_start,y_cropped_end+1)
        # do the cropping 
        I_DPC_cropped[y_idx_cropped,x_idx_cropped] = I_dpc[y_idx_FOV,x_idx_FOV]
        I_fluorescence_cropped[y_idx_cropped,x_idx_cropped,:] = I_fluorescence[y_idx_FOV,x_idx_FOV,:]
        
        # combine
        if counter == 0:
            I = np.dstack((I_fluorescence_cropped,I_DPC_cropped))[np.newaxis,:]
            if generate_separate_images:
                I_DAPI = I_fluorescence_cropped[np.newaxis,:]
                I_DPC = I_DPC_cropped[np.newaxis,:]
        else:
            I = np.concatenate((I,np.dstack((I_fluorescence_cropped,I_DPC_cropped))[np.newaxis,:]))
            if generate_separate_images:
                I_DAPI = np.concatenate((I_DAPI,I_fluorescence_cropped[np.newaxis,:]))
                I_DPC = np.concatenate((I_DPC,I_DPC_cropped[np.newaxis,:]))
        counter = counter + 1

    if counter == 0:
        print('no spot in this FOV')
    else:
        # gcs
        if settings['save to gcs']:
            fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])
            dir_out = settings['bucket_destination'] + '/' + settings['dataset_id'] + '/' + 'spot_images_fov'

        # convert to xarray
        # data = xr.DataArray(I,coords={'c':['B','G','R','DPC']},dims=['t','y','x','c'])
        data = xr.DataArray(I,dims=['t','y','x','c'])
        data = data.expand_dims('z')
        data = data.transpose('t','c','z','y','x')
        data = (data*255).astype('uint8')
        ds = xr.Dataset({'spot_images':data})
        # ds.spot_images.data = (ds.spot_images.data*255).astype('uint8')
        if settings['save to gcs']:
            store = fs.get_mapper(dir_out + '/' + str(i) + '_' + str(j) + '.zarr')
        else:
            store = dir_out + '/' + str(i) + '_' + str(j) + '.zarr'
        ds.to_zarr(store, mode='w')

        if generate_separate_images:
            
            data = xr.DataArray(I_DAPI,dims=['t','y','x','c'])
            data = data.expand_dims('z')
            data = data.transpose('t','c','z','y','x')
            data = (data*255).astype('uint8')
            ds = xr.Dataset({'spot_images':data})
            if settings['save to gcs']:
                store = fs.get_mapper(dir_out + '/' + str(i) + '_' + str(j) + '_fluorescence.zarr')
            else:
                store = dir_out + '/' + str(i) + '_' + str(j) + '_fluorescence.zarr'
            ds.to_zarr(store, mode='w')

            data = xr.DataArray(I_DPC,dims=['t','y','x'])
            data = data.expand_dims(('z','c'))
            data = data.transpose('t','c','z','y','x')
            data = (data*255).astype('uint8')
            ds = xr.Dataset({'spot_images':data})
            if settings['save to gcs']:
                store = fs.get_mapper(dir_out + '/' + str(i) + '_' + str(j) + '_DPC.zarr')
            else:
                store = dir_out + '/' + str(i) + '_' + str(j) + '_DPC.zarr'
            ds.to_zarr(store, mode='w')
