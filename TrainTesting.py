import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import torchmetrics as tm
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, dl):
    global device

    model.eval()
    val_acc = tm.Accuracy(task='binary', average='micro').to(device)
    val_bce = nn.BCELoss()
    val_loss_hist = []

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out = model(x)
            val_acc.update(out, y)
            val_loss = val_bce(out, y)
            val_loss_hist.append(val_loss.item())

    return val_acc.compute().item(), np.mean(val_loss_hist)


def train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion):
    global device
    
    pbar = tqdm(total=epochs*len(train_dl))
    model.train()
    train_acc = tm.Accuracy(task='binary', average='micro').to(device)
    last_val_acc = -1

    val_acc_hist = []
    train_acc_hist = []

    val_loss_hist = []
    train_loss_hist = []
    train_acc 
    for epoch in range(epochs):
        local_train_acc_hist = []
        local_train_loss_hist = []
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc.update(out, y)
            local_train_acc_hist.append(train_acc.compute().item())
            local_train_loss_hist.append(loss.item())
            pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {local_train_acc_hist[-1]:.4f}, val_acc (previous): {last_val_acc:.4f} | best_val_acc: {max(val_acc_hist) if len(val_acc_hist) > 0 else -1:.4f} at epoch {np.argmax(val_acc_hist)+1 if len(val_acc_hist) > 0 else -1}')
            pbar.update(1)
            
        train_loss_hist.append(np.mean(local_train_loss_hist))
        train_acc_hist.append(np.mean(local_train_acc_hist))

        last_val_acc, last_val_loss = validate(model, val_dl)
        val_acc_hist.append(last_val_acc)
        val_loss_hist.append(last_val_loss)
    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist
