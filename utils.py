import imageio
import cv2
# import cupy as cp # conda install -c conda-forge cupy==10.2
# import cupyx.scipy.ndimage
import numpy as np
from scipy import signal
import pandas as pd
import xarray as xr
import gcsfs

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
