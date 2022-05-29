import copy
from typing import Tuple, Union, List
import numpy as np
from tqdm import tqdm as tqdm
from sklearn.metrics import r2_score

import torch

from .utils import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from IPython.display import clear_output


def test(model, loader, loss_fn, device) -> Tuple[float, float]:
    """ returns average loss and score
    """
    model.eval()

    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for (X, y) in loader:
            y_gpu = y.to(device)
            out = model(X)

            loss = loss_fn(out, y_gpu)
            total_loss += loss.item()

            y_true.extend(y)
            y_pred.extend(out.detach().cpu())
    
    return total_loss / len(loader), r2_score(y_true, y_pred)


def train(model, train_loader, val_loader, loss_fn,
          optimizer, device, scheduler=None, num_epochs=10, plot=True, plot_update_every=5):
    """ returns best model on validation
    """
    train_losses = []
    train_scores = []
    val_losses = []
    val_scores = []
    
    best_val_score = -torch.inf
    best_model = None

    for epoch in range(num_epochs + 1):
        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        for i_step, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_gpu = y.to(device)
            out = model(X)

            loss = loss_fn(out, y_gpu)
            loss.backward()
            optimizer.step()

            y_true.extend(y)
            y_pred.extend(out.detach().cpu())

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        if scheduler is not None:
            scheduler.step()

        train_scores.append(r2_score(y_true, y_pred))
        val_loss, val_score = test(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        val_scores.append(val_score)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model)

        if plot and epoch > 0 and epoch % plot_update_every == 0:
            clear_output(True)
            _, axes = plt.subplots(1, 2, figsize=(20, 6))
            
            sns.lineplot(ax=axes[0], x=range(epoch + 1), y=train_losses, label='Train', color='blue')
            sns.lineplot(ax=axes[0], x=range(epoch + 1), y=val_losses, label='Val', color='red')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()

            sns.lineplot(ax=axes[1], x=range(epoch + 1), y=val_scores, label='Val', color='red')
            sns.lineplot(ax=axes[1], x=range(epoch + 1), y=train_scores, label='Train', color='blue')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Score')
            axes[1].legend()

            plt.show()
            # print(f'Epoch {epoch}, Loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}, Val R2: {val_scores[-1]:.4f}')
    
    return best_model