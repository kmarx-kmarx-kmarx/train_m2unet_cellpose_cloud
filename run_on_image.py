import os
from statistics import mode
from interactive_m2unet import M2UnetInteractiveModel

import numpy as np
import imageio
import albumentations as A
from skimage.filters import threshold_otsu
from skimage.measure import label   
from tqdm import tqdm
import time
import cv2
import random as rng
# specify the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

def main():
    # setting up
    torch.cuda.empty_cache()
    data_dir = '/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/data'#'./Cellpose Exports for 30deg 4ul 15mmps' # data should contain a train and a test folder
    model_root = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/m2unet_model"
    resume = True
    corrid = "200"
    nn = 14 # 14 best
    pretrained_model = None # os.path.join(model_root, str(corrid), "model.h5")

    # define the transforms
    transform = A.Compose(
        [
            A.RandomCrop(1500, 1500),
            A.Rotate(limit=(-10, 10), p=1),
            A.HorizontalFlip(p=0.5),
            A.CenterCrop(1024, 1024),  
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
            A.RandomToneCurve (scale=0.05, always_apply=False, p=0.5),
        ]
    )
    # unet model hyperparamer can be found here: https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=f899f7a8a9144b3f946c4a1362f7e38ae0c00c59&device=unknown_device&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f79696e676b61697368612f6b657261732d756e65742d636f6c6c656374696f6e2f663839396637613861393134346233663934366334613133363266376533386165306330306335392f6578616d706c65732f757365725f67756964655f6d6f64656c732e6970796e62&logged_in=true&nwo=yingkaisha%2Fkeras-unet-collection&path=examples%2Fuser_guide_models.ipynb&platform=mac&repository_id=323426984&repository_type=Repository&version=95#Swin-UNET
    model_config = {
        "type": "m2unet",
        "activation": "sigmoid",
        "output_channels": 1,
        "loss": {"name": "BCELoss", "kwargs": {}},
        "optimizer": {"name": "RMSprop", "kwargs": {"lr": 1e-2, "weight_decay": 1e-8, "momentum": 0.9}},
        "augmentation": A.to_dict(transform),
    }

    perform_testing(nn, data_dir, model_config, model_root, resume, pretrained_model, corrid)


# test
def perform_testing(nn, data_dir, model_config, model_root, resume, pretrained_model, corrid):
    sz = 1024
    offset = 1
    lft = cv2.imread(os.path.join(data_dir, '0_1_0_BF_LED_matrix_left_half.bmp'))[(offset*sz):((offset+1)*sz),(offset*sz):((offset+1)*sz),0]
    rht = cv2.imread(os.path.join(data_dir, '0_1_0_BF_LED_matrix_right_half.bmp'))[(offset*sz):((offset+1)*sz),(offset*sz):((offset+1)*sz),0]
    flatfield_left = np.load('flatfield_left.npy')
    flatfield_right = np.load('flatfield_right.npy')
    lft = lft.astype('float')/255
    rht = rht.astype('float')/255
    lft = lft/flatfield_left[:sz,:sz]
    rht = rht/flatfield_right[:sz,:sz]

    model = M2UnetInteractiveModel(
        model_config=model_config,
        model_dir=model_root,
        resume=resume,
        pretrained_model=pretrained_model,
        default_save_path=os.path.join(model_root, f'{corrid}_model_{nn}.pth'),
    )
    os.makedirs(f"m2unet_{nn}", exist_ok=True)
    img = generate_dpc(lft, rht)
    imageio.imwrite(f"m2unet_{nn}/octopi-inputs.png", img)
    inputs = np.stack([img,]*3, axis=2)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = (inputs - np.mean(inputs)) /np.std(inputs)
    for i in range(5):
        t0 = time.time()
        results = model.predict(inputs)
        output = np.clip(results[0] * 255, 0, 255)[:, :, 0].astype('uint8')
        threshold = threshold_otsu(output)
        mask = ((output > threshold) * 255).astype('uint8')
        # predict_labels = label(mask)
        print(time.time()-t0)
        imageio.imwrite(f"m2unet_{nn}/octopi-pred-prob.png", output)
        #imageio.imwrite(f"m2unet_{nn}/octopi-pred-labels.png", predict_labels)
        imageio.imwrite(f"m2unet_{nn}/octopi-pred-mask_otsu.png", mask)
        #watershed
        # sharpen the image - get boundaries
        t0 = time.time()
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        imgLaplacian = cv2.filter2D(output, cv2.CV_32F, kernel)
        _, imgLaplacian = cv2.threshold(imgLaplacian, 50, 255, cv2.THRESH_BINARY)
        sharp = np.float32(output)
        imgResult = sharp - imgLaplacian

        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype('uint8')
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)

        #imageio.imwrite(f"m2unet_{nn}/octopi-pred-mask_sharp.png", imgLaplacian)
        _, bw = cv2.threshold(imgResult, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #imageio.imwrite(f"m2unet_{nn}/octopi-pred-mask_binary-sharp.png", bw)
        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
        kernel1 = np.ones((2,2), dtype=np.uint8)
        # first, open to remove noise
        #dist = cv2.morphologyEx(dist,cv2.MORPH_OPEN,kernel1, iterations = 1)
        dist = cv2.dilate(dist, kernel1, iterations = 2)
        #imageio.imwrite(f"m2unet_{nn}/octopi-pred-mask_exp.png", dist)
        dist_8u = dist.astype('uint8')
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers = np.zeros(dist.shape, dtype=np.int32)
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i+1), -1)
        imgResult = cv2.cvtColor(imgResult,cv2.COLOR_GRAY2RGB)
        cv2.watershed(imgResult, markers)
        mark = markers.astype('uint8')
        mark = cv2.bitwise_not(mark)
        colors = []
        for contour in contours:
            colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
        dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i,j]
                if index > 0 and index <= len(contours):
                    dst[i,j,:] = colors[index-1]
        print(time.time()-t0)
        imageio.imwrite(f"m2unet_{nn}/octopi-pred-watershed.png", dst)
    print("all done")

def generate_dpc(I_1,I_2):
    I_dpc = np.divide(I_1-I_2,I_1+I_2)
    I_dpc = I_dpc + 0.5
    I_dpc[I_dpc<0] = 0
    I_dpc[I_dpc>1] = 1

    I_dpc = (255*I_dpc)

    return I_dpc.astype('uint8')

if __name__ == "__main__":
    main()