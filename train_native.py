import os
from statistics import mode
from interactive_m2unet import M2UnetInteractiveModel

import numpy as np
import imageio
import albumentations as A
from skimage.filters import threshold_otsu
from skimage.measure import label   
from tqdm import tqdm
import csv
import time
import cv2
# specify the gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
# 14 best
def main():
    # setting up
    torch.cuda.empty_cache()
    data_dir = '/home/prakashlab/Documents/cellpose_img/PF Tanzania full (U3D_201910_2022-01-11_23-11-36.799392)'#'./Cellpose Exports for 30deg 4ul 15mmps' # data should contain a train and a test folder
    model_root = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/m2unet_model"
    epochs = 8000
    steps = 1
    resume = True
    corrid = "200"
    nn = 52
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

    #perform_training(data_dir, model_root, epochs, steps, resume, corrid, pretrained_model, transform, model_config)
    perform_testing(nn, data_dir, model_config, model_root, resume, pretrained_model, corrid)


def perform_training(data_dir, model_root, epochs, steps, resume, corrid, pretrained_model, transform, model_config):
    os.makedirs(os.path.join(model_root, str(corrid)), exist_ok=True)
    A.save(transform, model_root + "/transform.json")

    # check if GPU is available
    print(f'GPU: {torch.cuda.is_available()}')

    model = M2UnetInteractiveModel(
        model_config=model_config,
        model_dir=model_root,
        resume=resume,
        pretrained_model=pretrained_model,
        default_save_path=os.path.join(model_root, str(corrid), "model.pth"),
    )
    # load samples
    train_samples = load_samples(data_dir + '/train')
    # train the model 
    iterations = 0
    for epoch in tqdm(range(epochs)):
        losses = []
        # image shape: 512, 512, 3
        # labels shape: 512, 512, 1
        for (image, labels) in train_samples:
            mask = model.transform_labels(labels)
            x = np.expand_dims(image, axis=0)
            x = (x - np.mean(x)) /np.std(x)
            y = np.expand_dims(mask, axis=0)
            losses = []
            for _ in range(steps):
                # x and y will be augmented for each step
                loss = model.train_on_batch(x, y)
                losses.append(loss)
                iterations += 1
                #print(f"iteration: {iterations}, loss: {loss}")
    model.save()

# test
def perform_testing(num_models, data_dir, model_config, model_root, resume, pretrained_model, corrid):
    test_samples = load_samples(data_dir)
    with open("results.csv", 'w') as f:
        write = csv.writer(f)
        for nn in range(num_models):
            model = M2UnetInteractiveModel(
                model_config=model_config,
                model_dir=model_root,
                resume=resume,
                pretrained_model=pretrained_model,
                default_save_path=os.path.join(model_root, f'200_model_{nn}.pth'),
            )
            os.makedirs(f"m2unet_{nn}", exist_ok=True)
            row = [nn]
            jrow = []
            jrow2 = []
            jrow3 = []
            for i, sample in enumerate(test_samples):
                inputs = sample[0].astype("float32")[None, :1024, :1024, :]
                #imageio.imwrite(f"m2unet_{nn}/octopi-inputs_{i}.png", inputs[0].astype('uint8'))
                inputs = (inputs - np.mean(inputs)) /np.std(inputs)
                #labels = sample[1].astype("float32")[None, :1024, :1024, :] * 255
                #imageio.imwrite(f"m2unet_{nn}/octopi-labels_{i}.png", labels[0].astype('uint8'))
                t0 = time.time()
                results = model.predict(inputs)
                output = np.clip(results[0] * 255, 0, 255)[:, :, 0].astype('uint8')
                threshold = threshold_otsu(output)
                mask = ((output > threshold) * 255).astype('uint8')
                #predict_labels = label(mask)
                print(time.time()-t0)
                #imageio.imwrite(f"m2unet_{nn}/octopi-pred-prob_{i}.png", output)
                #imageio.imwrite(f"m2unet_{nn}/octopi-pred-labels_{i}.png", predict_labels)
                imageio.imwrite(f"m2unet_{nn}/octopi-pred-mask_otsu_{i}.png", mask)
                #watershed
                o2 = cv2.cvtColor(output,cv2.COLOR_GRAY2RGB)
                kernel = np.ones((3,3),np.uint8)
                opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
                sure_bg = cv2.dilate(opening,kernel,iterations=3)
                dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)
                ret, markers = cv2.connectedComponents(sure_fg)
                markers = markers+1
                markers[unknown==255] = 0
                markers = cv2.watershed(o2,markers)
                o2[markers == -1] = [0,0,0]
                o3 = np.sum(o2, axis=2)
                #o2[o3 > 0] = [1,1,1]
                o2 = o2[:,:,0]
                imageio.imwrite(f"m2unet_{nn}/octopi-pred-mask_watershed_{i}.png", o2*255)
                # calculate jaccard similarity
                j = jaccard_sim(results[0,:,:,0], sample[1].astype("float32")[None, :1024, :1024, 0])
                jrow.append(j)
                j = jaccard_sim(mask/255, sample[1].astype("float32")[None, :1024, :1024, 0])
                jrow2.append(j)
                jrow3.append(jaccard_sim(o2, sample[1].astype("float32")[None, :1024, :1024, 0]))
            row.append(np.mean(jrow))
            row.append(np.mean(jrow2))
            row.append(np.mean(jrow3))
            write.writerow(row)

    print("all done")


def jaccard_sim(img1, img2):
    n = np.prod(img1.shape)
    a = img1 * img2
    b = img1 + img2 - a
    J = a/b
    J[np.isnan(J)] = 1
    j = np.sum(J)/n

    return j

# a function for loading cellpose output (image, mask and outline)
def load_samples(train_dir):
    npy_files = [os.path.join(train_dir, s) for s in os.listdir(train_dir) if s.endswith('.npy')]
    samples = []
    for file in npy_files:
        print(file)
        items = np.load(file, allow_pickle=True).item()
        mask = (items['masks'][:, :, None]  > 0) * 1.0
        outline = (items['outlines'][:, :, None]  > 0) * 1.0
        mask = mask * (1.0 - outline)
        sample = (np.stack([items['img'],]*3, axis=2), mask)
        samples.append(sample)
    return samples

if __name__ == "__main__":
    main()