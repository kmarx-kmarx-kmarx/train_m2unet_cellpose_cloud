# Train M2U-Net model using raw data in one bucket and corresponding masks in another bucket

def main():
    cellpose_model_path = '/home/prakashlab/Documents/kmarx/tb_segment/pipeline/cellpose_residual_on_style_on_concatenation_off_train_2023_02_10_10_52_15.693300_epoch_4001'
    # gcs data source
    gcs_project = 'soe-octopi'
    gcs_token = '/home/prakashlab/Documents/kmarx/data-20220317-keys.json' 
    bucket_data = ''
    bucket_masks = ''
    datasets = []

    save_interval = 50
    n_views_train = 101 # number of views from each dataset to train on
    n_views_valid = 50  # number of views from each dataset to run validation on
                        # make (n_views_valid + n_views_train) a prime number for good cross-validation
    random_views = True
    random.seed(0)
    # illumination correction
    flatfield_left = np.load('/home/prakashlab/Documents/kmarx/tb_segment/pipeline/illumination correction/flatfield_left.npy')
    flatfield_right = np.load('/home/prakashlab/Documents/kmarx/tb_segment/pipeline/illumination correction/flatfield_right.npy')
    # m2unet training
    epochs = 30
    sz = 1024
    model_root = ""
    transform = A.Compose(
        [
            A.Rotate(limit=(-90, 90), p=1), # set to -10, 10 for dpc
            A.RandomCrop(1500, 1500),
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

    # initialize google cloud
    
    # Get a list of all the images on google cloud images
    aq_params = {}
    for dataset_id in datasets:
        json_file = fs.cat(os.path.join(root_dir, exp_id, "acquisition parameters.json"))
        acquisition_params = json.loads(json_file)
        aq_params[dataset_id] = [acquisition_params['Ny'], acquisition_params['Nx']]

    return

if __name__ == "__main__":
    main()