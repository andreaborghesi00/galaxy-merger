import os
import numpy as np
from tqdm import tqdm

import galaxy_dataset

from models.resnet18 import ResNet18
from models.fast_heavy_cnn import FastHeavyCNN
from models.heavy_cnn import HeavyCNN
from models.deepmerge import DeepMerge

import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# note: the first time you run this script, it will download two datasets (pristine and noisy)
# this will take a while, but the datasets will be saved locally for future runs.
# also, the first time you run this script, denoised datasets will be generated and saved locally for the same reason
# while doing so it might crash if you don't have enough memory, so be careful, and if it crashes, just run it again, as
# the datasets will be saved and the script will skip the generation part

if __name__ == "__main__":
    seed = 42 # for reproducibility, change it to None if you want random results
    dataset_types = ["pristine", "noisy", "fft", "bg_sub", "top_hat", "unet"] 
    models = [ResNet18, FastHeavyCNN, DeepMerge]
    models_names = [model.__name__ for model in models]

    models_root = os.path.join(utils.RES_DIR, "models")
    run_no = 0 # run number
    dict_models_data = { model.__name__ : { dataset_type : os.path.join(models_root, model.__name__, f"{utils.PREFIX_MODELS}_TEST_{model.__name__}_{dataset_type}_{run_no}.pth") for dataset_type in dataset_types } for model in models}
    dict_data_models = { dataset_type : { model.__name__ : os.path.join(models_root, model.__name__, f"{utils.PREFIX_MODELS}_TEST_{model.__name__}_{dataset_type}_{run_no}.pth") for model in models} for dataset_type in dataset_types}

    # accs = np.zeros((len(models), len(dataset_types)))
    inf_times = np.zeros((len(models), len(dataset_types)))
    for i, dataset_type in enumerate(dataset_types):
        for j, model_class in enumerate(models):
            experiment_name = f"{model_class.__name__}_{dataset_type}"

            X, y = galaxy_dataset.load_dataset(dataset_type=dataset_type)
        
            X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.67, stratify=y_valtest, random_state=seed)
            test_dl = galaxy_dataset.get_dataloader(X_test, y_test, None, batch_size=256, num_workers=4, shuffle=False)
            
            utils.combined_pr_auc_compare_models(models, 
                                                 list(dict_data_models[dataset_type].values()), 
                                                 test_dl, 
                                                 dataset_type, 
                                                 file_name=f"{utils.PREFIX_ROC}_combined_{dataset_type}_{run_no}")
            
            # model = Models.load_model(model_class, dict_data_models[dataset_type][model_class.__name__], last=False, verbose=False).to(TrainTesting.device)
            # Utils.compute_confusion_matrix(model, test_dl, dataset_type, experiment_name)
            # acc, loss, inf_time =  TrainTesting.validate(model, test_dl)
            # inf_times[j, i] = inf_time
            # accs[j, i] = acc
            # print(f"Model: {model.__class__.__name__}, Dataset: {dataset_type}, Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}, FPS: {1/inf_time:.3f}")
    
    print(np.mean(inf_times, axis=1))

    # np.save('results/test_accuracy_table.npy', accs)
    for model in tqdm(models):
        utils.combined_roc_auc_compare_datasets(model,
                                                list(dict_models_data[model.__name__].values()), 
                                                dataset_types, 
                                                file_name=f"{utils.PREFIX_ROC}_combined_{model.__name__}_{run_no}")
