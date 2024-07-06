from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch
import TrainTesting
import seaborn as sns
from sklearn.metrics import roc_curve, auc

RES_DIR = "results"
PREFIX_MODELS = "history"
PREFIX_PLOTS = "loss_acc"
PREFIX_CONF = "confusion_matrix"
PREFIX_ROC = "roc"


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
    global RES_DIR, PREFIX_PLOTS, PREFIX_CONF
    plot_dir = os.path.join(RES_DIR, "plots", model.__class__.__name__)
    conf_dir = os.path.join(RES_DIR, "confusion_matrices", model.__class__.__name__)
    os.makedirs(plot_dir, exist_ok=True)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(RES_DIR, f'{PREFIX_PLOTS}_{experiment_name}_{i}.pkl')): i += 1

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
    # confusion matrix
    y_true, y_pred = _test_probas(model, test_dl)
    y_pred = (y_pred > 0.5).astype(int) # thresholding, since the model returns probabilities
    cm = confusion_matrix(y_true, y_pred)

    # sns heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True, xticklabels=['Non-Merger', 'Merger'], yticklabels=['Non-Merger', 'Merger'])
    plt.xlabel('Predicted')
    plt.ylabel('True')   
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(conf_dir, f'{PREFIX_CONF}_{experiment_name}_{i}.png'))
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
    while os.path.exists(os.path.join(RES_DIR, f'{PREFIX_ROC}_{experiment_name}_{i}.pkl')): i += 1

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
    return roc_auc


def combined_roc_auc(models, test_dl, experiment_names, save_path=None):
    """
    Calculate and plot the combined ROC curve and AUC for multiple models.

    Parameters:
    - models (list): A list of trained models.
    - test_dl (DataLoader): The test data loader.
    - experiment_names (list): A list of names for each experiment.
    - save_path (str, optional): The file path to save the plot. Defaults to None.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    for model, experiment_name in zip(models, experiment_names):
        y_true, y_pred = _test_probas(model, test_dl)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{experiment_name} (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)


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

    i = 0
    while os.path.exists(os.path.join(model_dir, f'{PREFIX_MODELS}_{experiment_name}_{i}.pkl')): i += 1
    
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
