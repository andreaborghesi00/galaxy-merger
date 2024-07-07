import os
import numpy as np
import gc
from tqdm import tqdm

import GalaxyDataset
import TrainTesting
import Models
import Utils

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import albumentations as albus
from albumentations.pytorch import ToTensorV2

# note: the first time you run this script, it will download two datasets (pristine and noisy)
# this will take a while, but the datasets will be saved locally for future runs.
# also, the first time you run this script, denoised datasets will be generated and saved locally for the same reason
# while doing so it might crash if you don't have enough memory, so be careful, and if it crashes, just run it again, as
# the datasets will be saved and the script will skip the generation part

if __name__ == "__main__":
    seed = 42 # for reproducibility, change it to None if you want random results
    dataset_types = ["pristine", "noisy", "fft", "bg_sub", "top_hat", "unet"] 
    models = [Models.ResNet18, Models.FastHeavyCNN, Models.DeepMerge]

    models_root = os.path.join(Utils.RES_DIR, "models")
    run_no = 0 # run number
    dict_models_data = { model.__name__ : { dataset_type : os.path.join(models_root, model.__name__, f"{Utils.PREFIX_MODELS}_TEST_{model.__name__}_{dataset_type}_{run_no}.pth") for dataset_type in dataset_types } for model in models}
    dict_data_models = { dataset_type : { model.__name__ : os.path.join(models_root, model.__name__, f"{Utils.PREFIX_MODELS}_TEST_{model.__name__}_{dataset_type}_{run_no}.pth") for model in models} for dataset_type in dataset_types}

    for dataset_type in dataset_types:
        for model_class in models:
            X, y = GalaxyDataset.load_dataset(dataset_type=dataset_type)
        
            X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.67, stratify=y_valtest, random_state=seed)
            test_dl = GalaxyDataset.get_dataloader(X_test, y_test, None, batch_size=256, num_workers=4, shuffle=False)
            
            # Utils.combined_roc_auc_compare_models(models, list(dict_data_models[dataset_type].values()), test_dl, dataset_type, file_name=f"{Utils.PREFIX_ROC}_combined_{dataset_type}_{run_no}")
            
            model = Models.load_model(model_class, dict_data_models[dataset_type][model_class.__name__], last=False, verbose=False).to(TrainTesting.device)
            acc, loss, inf_time =  TrainTesting.validate(model, test_dl)
            print(f"Model: {model.__class__.__name__}, Dataset: {dataset_type}, Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}, FPS: {1/inf_time:.3f}")
        
    # for model in tqdm(models):
    #     Utils.combined_roc_auc_compare_datasets(model, list(dict_models_data[model.__name__].values()), dataset_types, file_name=f"{Utils.PREFIX_ROC}_combined_{model.__name__}_{run_no}")
