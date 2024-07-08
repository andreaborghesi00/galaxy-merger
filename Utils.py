from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch
import TrainTesting
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import Models
import TrainTesting
import GalaxyDataset

RES_DIR = "results"
PREFIX_MODELS = "history"
PREFIX_PLOTS = "loss_acc"
PREFIX_CM = "confusion_matrix"
PREFIX_ROC = "roc"
PREFIX_PR = "pr"


def plots(model, experiment_name, test_dl, train_loss, val_loss, train_acc, val_acc):
    """
    Generate and save plots for loss, accuracy, and confusion matrix.

    Args:
        model (object): The trained model object.
        experiment_name (str): The name of the experiment.
        test_dl (object): The test data loader object.
        train_loss (list): List of training loss values.
        val_loss (list): List of validation loss values.
        train_acc (list): List of training accuracy values.
        val_acc (list): List of validation accuracy values.
    """
    global RES_DIR, PREFIX_PLOTS
    plot_dir = os.path.join(RES_DIR, "plots", model.__class__.__name__)
    os.makedirs(plot_dir, exist_ok=True)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(plot_dir, f'{PREFIX_PLOTS}_{experiment_name}_{i}.png')): i += 1

    # loss and accuracy plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'loss_acc_{experiment_name}_{i}.png'))
    plt.close('all')

def compute_confusion_matrix(model, test_dl, experiment_name, normalized=True):
    global RES_DIR, PREFIX_CM
    conf_dir = os.path.join(RES_DIR, "confusion_matrices", model.__class__.__name__)
    os.makedirs(conf_dir, exist_ok=True)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(conf_dir, f'{PREFIX_CM}_{experiment_name}_{i}.png')): i += 1

    y_true, y_pred = _test_probas(model, test_dl)
    y_pred = (y_pred > 0.5).astype(int) # thresholding, since the model returns probabilities
    cm = confusion_matrix(y_true, y_pred)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # sns heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', cbar=True, square=True, xticklabels=['Non-Merger', 'Merger'], yticklabels=['Non-Merger', 'Merger'])
    plt.xlabel('Predicted')
    plt.ylabel('True')   
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(conf_dir, f'{PREFIX_CM + (f"_normalized" if normalized else "")}_{experiment_name}_{i}.png'))
    plt.close('all')

def _test_probas(model, test_dl):
    """
    Compute the true and predicted values for a given model and test dataloader.

    Args:
        model (torch.nn.Module): The trained model.
        test_dl (torch.utils.data.DataLoader): The test dataloader.

    Returns:
        tuple: A tuple containing the true values and predicted values.
    """
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(TrainTesting.device)
            y = y.to(TrainTesting.device)
            out = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_true,y_pred


def compute_roc_auc(model, test_dl, experiment_name, save_path=None):
    """
    Compute the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) for a given model.

    Args:
        model (object): The trained model.
        test_dl (DataLoader): The DataLoader object containing the test data.
        experiment_name (str): The name of the experiment.
        save_path (str, optional): The path to save the ROC curve plot. Defaults to None.

    Returns:
        float: The computed ROC AUC score.
    """
    global RES_DIR, PREFIX_ROC
    roc_dir = os.path.join(RES_DIR, "roc_auc", model.__class__.__name__)
    os.makedirs(roc_dir, exist_ok=True)

    y_true, y_pred = _test_probas(model, test_dl)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(roc_dir, f'{PREFIX_ROC}_{experiment_name}_{i}.png')): i += 1

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.savefig(os.path.join(roc_dir, f'{PREFIX_ROC}_{experiment_name}_{i}.png'))
    plt.close('all')
    
    return roc_auc

def compute_precision_recall(model, test_dl, experiment_name, save_path=None):
    """
    Compute the Precision-Recall curve for a given model.

    Args:
        model (object): The trained model.
        test_dl (DataLoader): The DataLoader object containing the test data.
        experiment_name (str): The name of the experiment.
        save_path (str, optional): The path to save the Precision-Recall curve plot. Defaults to None.

    Returns:
        None
    """
    global RES_DIR, PREFIX_PR
    pr_dir = os.path.join(RES_DIR, "precision_recall", model.__class__.__name__)
    os.makedirs(pr_dir, exist_ok=True)

    y_true, y_pred = _test_probas(model, test_dl)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(pr_dir, f'{PREFIX_PR}_{experiment_name}_{i}.png')): i += 1

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='Precision Recall curve (area = %0.2f)' % pr_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.savefig(os.path.join(pr_dir, f'{PREFIX_PR}_{experiment_name}_{i}.png'))
    plt.close('all')




def combined_roc_auc_compare_models(model_classes, model_paths, test_dl, dataset_type, file_name=None):
    global RES_DIR, PREFIX_ROC

    roc_combined_dir = os.path.join(RES_DIR, "roc_auc", "combined")
    os.makedirs(roc_combined_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for idx in range(len(model_classes)):
        model = Models.load_model(model_classes[idx], model_paths[idx]).to(TrainTesting.device)
        y_true, y_pred = _test_probas(model, test_dl)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model.__class__.__name__} (area = {roc_auc:.2f})', alpha=0.7)
        torch.cuda.empty_cache()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {dataset_type} dataset')
    plt.legend(loc="lower right")
    if file_name:
        plt.savefig(os.path.join(roc_combined_dir, f"{file_name}.png"))
    else:
        i = 0 # i know it's ugly, leave me alone
        while os.path.exists(os.path.join(roc_combined_dir, f'{PREFIX_ROC}_combined_{i}.png')): i += 1
        plt.savefig(os.path.join(roc_combined_dir, f'{PREFIX_ROC}_combined_{i}.png'))
    plt.close('all')


def combined_roc_auc_compare_datasets(model_class, model_paths, dataset_types, file_name=None, split_seed=0):
    global RES_DIR, PREFIX_ROC

    roc_combined_dir = os.path.join(RES_DIR, "roc_auc", "combined")
    os.makedirs(roc_combined_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for idx in range(len(dataset_types)):
        model = Models.load_model(model_class, model_paths[idx]).to(TrainTesting.device) 
        X, y = GalaxyDataset.load_dataset(dataset_type=dataset_types[idx])
        X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.67, stratify=y_valtest, random_state=split_seed)
        test_dl = GalaxyDataset.get_dataloader(X_test, y_test, None, batch_size=256, num_workers=4, shuffle=False)

        y_true, y_pred = _test_probas(model, test_dl)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{dataset_types[idx]} (area = {roc_auc:.2f})', alpha=0.7)
        torch.cuda.empty_cache()

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_class.__name__}')
    plt.legend(loc="lower right")
    if file_name:
        plt.savefig(os.path.join(roc_combined_dir, f"{file_name}.png"))
    else:
        i = 0 # i know it's ugly, leave me alone
        while os.path.exists(os.path.join(roc_combined_dir, f'{PREFIX_ROC}_combined_{i}.png')): i += 1
        plt.savefig(os.path.join(roc_combined_dir, f'{PREFIX_ROC}_combined_{i}.png'))
    plt.close('all')

def combined_pr_auc_compare_models(model_classes, model_paths, test_dl, dataset_type, file_name=None):
    global RES_DIR, PREFIX_PR

    pr_combined_dir = os.path.join(RES_DIR, "pr_auc", "combined")
    os.makedirs(pr_combined_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for idx in range(len(model_classes)):
        model = Models.load_model(model_classes[idx], model_paths[idx]).to(TrainTesting.device)
        y_true, y_pred = _test_probas(model, test_dl)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
        plt.plot(recall, precision, lw=2, label=f'{model.__class__.__name__} (AP = {pr_auc:.2f})', alpha=0.7)
        torch.cuda.empty_cache()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves for {dataset_type} dataset')
    plt.legend(loc="lower left")
    if file_name:
        plt.savefig(os.path.join(pr_combined_dir, f"{file_name}.png"))
    else:
        i = 0 # i know it's ugly, leave me alone
        while os.path.exists(os.path.join(pr_combined_dir, f'{PREFIX_PR}_combined_{i}.png')): i += 1
        plt.savefig(os.path.join(pr_combined_dir, f'{PREFIX_PR}_combined_{i}.png'))
    plt.close('all')

def combined_pr_auc_compare_datasets(model_class, model_paths, dataset_types, file_name=None, split_seed=0):
    global RES_DIR, PREFIX_PR

    pr_combined_dir = os.path.join(RES_DIR, "pr_auc", "combined")
    os.makedirs(pr_combined_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for idx in range(len(dataset_types)):
        model = Models.load_model(model_class, model_paths[idx]).to(TrainTesting.device) 
        X, y = GalaxyDataset.load_dataset(dataset_type=dataset_types[idx])
        X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split_seed)
        X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.67, stratify=y_valtest, random_state=split_seed)
        test_dl = GalaxyDataset.get_dataloader(X_test, y_test, None, batch_size=256, num_workers=4, shuffle=False)

        y_true, y_pred = _test_probas(model, test_dl)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
        plt.plot(recall, precision, lw=2, label=f'{dataset_types[idx]} (AP = {pr_auc:.2f})', alpha=0.7)
        torch.cuda.empty_cache()

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves for {model_class.__name__}')
    plt.legend(loc="lower left")
    if file_name:
        plt.savefig(os.path.join(pr_combined_dir, f"{file_name}.png"))
    else:
        i = 0 # i know it's ugly, leave me alone
        while os.path.exists(os.path.join(pr_combined_dir, f'{PREFIX_PR}_combined_{i}.png')): i += 1
        plt.savefig(os.path.join(pr_combined_dir, f'{PREFIX_PR}_combined_{i}.png'))
    plt.close('all')

def save_results(model, experiment_name, train_loss, val_loss, train_acc, val_acc, optimizer, scheduler, best_state_dict):
    """
    Save the results of a model training experiment.

    Args:
        model (torch.nn.Module): The trained model.
        experiment_name (str): The name of the experiment.
        train_loss (float): The training loss.
        val_loss (float): The validation loss.
        train_acc (float): The training accuracy.
        val_acc (float): The validation accuracy.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler used for training.
        best_state_dict (dict): The state dictionary of the best model.

    Returns:
        None
    """
    global RES_DIR
    model_dir = os.path.join(RES_DIR, "models", model.__class__.__name__)
    os.makedirs(model_dir, exist_ok=True)

    history = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(model_dir, f'{PREFIX_MODELS}_{experiment_name}_{i}.pth')): i += 1

    torch.save({ # i know it seems overkill, but it takes a lot of time to train these models
        'last_model_state_dict': model.state_dict(),
        'best_model_state_dict': best_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model': model,
        'model_name': model.__class__.__name__,
        'optimizer': optimizer,
        'optimizer_name': optimizer.__class__.__name__,
        'scheduler': scheduler,
        'scheduler_name': scheduler.__class__.__name__,
        'history': history
    }, os.path.join(model_dir, f'{PREFIX_MODELS}_{experiment_name}_{i}.pth'))
