import os
import numpy as np
import gc

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


if __name__ == "__main__":
    dataset_types = ["pristine", "noisy", "fft", "bg_sub", "top_hat", "unet"]
    models = [Models.DeepMerge, Models.ResNet18, Models.HeavyCNN]
    
    for model_class in models:
        for dataset_type in dataset_types:
            experiment_name = f"TEST_{model_class.__name__}_{dataset_type}"
            print(f"Running experiment: {experiment_name}")
            # experiment_name = "test"

            # load dataset
            # dataset_type = "unet" # change here to the preferred dataset: noisy, pristine, fft, bg_sub, top_hat, gmm, unet
            X, y = GalaxyDataset.load_dataset(dataset_type=dataset_type)

            # split 70:10:20
            random_state = 42
            X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.67, random_state=random_state, stratify=y_valtest)

            # augmentations
            augmentations = albus.Compose([
                albus.HorizontalFlip(p=0.5),
                albus.VerticalFlip(p=0.5),
                albus.RandomRotate90(p=0.5),
                # albus.RandomBrightnessContrast(p=0.5), # they're already pretty noisy, so perhaps dont
                ToTensorV2()
            ])

            # create dataloaders
            batch_size = 256
            train_dl = GalaxyDataset.get_dataloader(X_train, y_train, batch_size=batch_size,num_workers=4, shuffle=True, transform=augmentations)
            val_dl = GalaxyDataset.get_dataloader(X_val, y_val, batch_size=batch_size,num_workers=4, shuffle=False)
            test_dl = GalaxyDataset.get_dataloader(X_test, y_test, batch_size=batch_size,num_workers=4, shuffle=False)
            del X_train, X_val, X_test, y_train, y_val, y_test, X, y

            # create model
            model = model_class().to(TrainTesting.device) # change here to the preferred model: ResNet18, HeavyCNN or DeepMerge
            criterion = nn.BCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, fused=True)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

            # train model
            epochs = 100
            train_loss, val_loss, train_acc, val_acc = TrainTesting.train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion)

            # save results
            Utils.save_results(model, experiment_name, train_loss, val_loss, train_acc, val_acc, optimizer, scheduler)

            # test model
            test_acc, test_loss = TrainTesting.validate(model, test_dl)
            print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

            # pretty plots
            Utils.plots(model, experiment_name, test_dl, train_loss, val_loss, train_acc, val_acc)

            del model, criterion, optimizer, scheduler, train_loss, val_loss, train_acc, val_acc, test_acc, test_loss
            torch.cuda.empty_cache()
            gc.collect()
            
