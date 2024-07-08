import gc

import GalaxyDataset
import TrainTesting
import Models
import Utils

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.model_selection import train_test_split

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

    for model_class in models:
        for dataset_type in dataset_types:
            experiment_name = f"{model_class.__name__}_{dataset_type}"
            print(f"################### Running experiment: {experiment_name} ###################")
            # experiment_name = "test"

            # load dataset
            # dataset_type = "unet" # change here to force the preferred dataset: noisy, pristine, fft, bg_sub, top_hat, gmm, unet
            X, y = GalaxyDataset.load_dataset(dataset_type=dataset_type)

            # split 70:10:20
            random_state = 42
            X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.67, random_state=random_state, stratify=y_valtest, random_state=seed)

            # augmentations
            augmentations = albus.Compose([
                albus.HorizontalFlip(p=0.5),
                albus.VerticalFlip(p=0.5),
                albus.RandomRotate90(p=0.5),
                # albus.RandomBrightnessContrast(p=0.5), # they're already pretty noisy, so perhaps dont
                ToTensorV2()
            ])

            # create dataloaders, if it crashes here, try reducing the batch size, remove pin_memory, prefetch_factor, persistent_workers, or num_workers
            batch_size = 256
            train_dl = GalaxyDataset.get_dataloader(X_train, y_train, augmentations, batch_size=batch_size,num_workers=4, shuffle=True, pin_memory=True, drop_last=False, prefetch_factor=1, persistent_workers=True)
            val_dl = GalaxyDataset.get_dataloader(X_val, y_val, None, batch_size=batch_size,num_workers=4, shuffle=False)
            test_dl = GalaxyDataset.get_dataloader(X_test, y_test, None, batch_size=batch_size,num_workers=4, shuffle=False)
            
            # weight for BCELoss, to balance the classes
            positive_samples = sum(y_train == 1)
            negative_samples = sum(y_train == 0)
            total_samples = positive_samples + negative_samples
            print(f"Positive samples: {positive_samples}, Negative samples: {negative_samples}, Total samples: {total_samples}")

            weight = negative_samples / positive_samples
            weight = torch.tensor([weight]).to(TrainTesting.device) 
            del X_train, X_val, X_test, y_train, y_val, y_test, X, y # memory's precious
            gc.collect()

            # create model
            model = model_class().to(TrainTesting.device) # change here to the preferred model: ResNet18, HeavyCNN or DeepMerge

            epochs = 150
            criterion = nn.BCELoss(weight=weight)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, fused=True)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            # train model
            train_loss, val_loss, train_acc, val_acc, best_state_dict = TrainTesting.train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion, validate_every=1, weight=weight)

            # save results
            Utils.save_results(model, experiment_name, train_loss, val_loss, train_acc, val_acc, optimizer, scheduler, best_state_dict)

            # test model
            test_acc, test_loss, _ = TrainTesting.validate(model, test_dl)
            print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')

            model.load_state_dict(best_state_dict) # load best model
            best_test_acc, best_test_loss, inference_time = TrainTesting.validate(model, test_dl)
            print(f'Best Test Accuracy: {best_test_acc:.4f}, Best Test Loss: {best_test_loss:.4f}, Inference Time: {inference_time:.6f} seconds')

            # pretty plots
            Utils.plots(model, experiment_name, test_dl, train_loss, val_loss, train_acc, val_acc)
            Utils.compute_confusion_matrix(model, test_dl, experiment_name, normalized=True)
            Utils.compute_roc_auc(model, test_dl, experiment_name)
            Utils.compute_precision_recall(model, test_dl, experiment_name)

            # clear memory, c'è la crisi
            del model, criterion, optimizer, scheduler, train_loss, val_loss, train_acc, val_acc, test_acc, test_loss
            torch.cuda.empty_cache()
            gc.collect()
            
