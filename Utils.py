from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch
import TrainTesting
import seaborn as sns

res_dir = "results"
plot_dir = os.path.join(res_dir, "plots")
def plots(model, experiment_name, test_dl, train_loss, val_loss, train_acc, val_acc):
    global res_dir
    os.makedirs(plot_dir, exist_ok=True)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(res_dir, f'history_{experiment_name}_{i}.pkl')): i += 1

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
    y_pred = (y_pred > 0.5).astype(int) # thresholding, since the model returns probabilities
    cm = confusion_matrix(y_true, y_pred)

    # sns heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True, xticklabels=['Non-Merger', 'Merger'], yticklabels=['Non-Merger', 'Merger'])
    plt.xlabel('Predicted')
    plt.ylabel('True')   
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plot_dir, f'confusion_matrix_{experiment_name}_{i}.png'))
    plt.close('all')
def save_results(model, experiment_name, train_loss, val_loss, train_acc, val_acc, optimizer, scheduler):
    os.makedirs(res_dir, exist_ok=True)

    history = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }

    i = 0
    while os.path.exists(os.path.join(res_dir, f'history_{experiment_name}_{i}.pkl')): i += 1
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model': model,
        'model_name': model.__class__.__name__,
        'optimizer': optimizer,
        'optimizer_name': optimizer.__class__.__name__,
        'scheduler': scheduler,
        'scheduler_name': scheduler.__class__.__name__,
        'history': history
    }, os.path.join(res_dir, f'history_{experiment_name}_{i}.pkl'))
