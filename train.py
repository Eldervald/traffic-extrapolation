import copy
from typing import Tuple, Union, List
import pandas as pd
import numpy as np
import networkx as nx
from tqdm.notebook import tqdm as tqdm
from sklearn.metrics import r2_score

# import node2vec

import torch

import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx

from utils import *
from dataset import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from IPython.display import clear_output


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device


def calc_score(pred, actual):
    return r2_score(actual, pred)


def test(model, loader, loss_fn) -> Tuple[float, float]:
    # returns average loss and score
    model.eval()

    scores = []
    total_loss = 0

    with torch.no_grad():
        for (X, y) in loader:
            X_gpu = X.to(device)
            y_gpu = y.to(device)
            out = model(X_gpu)
            scores.append(calc_score(out.detach().cpu(), y))
            loss = loss_fn(out, y_gpu)
            total_loss += loss.item()
    
    return total_loss / len(loader), np.mean(scores)


def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler=None, num_epochs=10, plotting=True):
    train_losses = []
    val_losses = []
    val_scores = []
    
    best_val_score = -torch.inf
    best_model = None

    for epoch in range(num_epochs + 1):
        model.train()
        total_loss = 0

        for i_step, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X_gpu = X.to(device)
            y_gpu = y.to(device)
            out = model(X_gpu)
            loss = loss_fn(out, y_gpu)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        if scheduler is not None:
            scheduler.step()

        val_loss, val_score = test(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        val_scores.append(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model)

        if plotting and epoch > 0:
            clear_output(True)
            _, axes = plt.subplots(1, 2, figsize=(20, 6))
            
            sns.lineplot(ax=axes[0], x=range(epoch + 1), y=train_losses, label='Train', color='blue')
            sns.lineplot(ax=axes[0], x=range(epoch + 1), y=val_losses, label='Val', color='red')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()

            sns.lineplot(ax=axes[1], x=range(epoch + 1), y=val_scores, label='Val', color='red')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Score')
            axes[1].legend()

            plt.show()
            # print(f'Epoch {epoch}, Loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}, Val R2: {val_scores[-1]:.4f}')
        
    return best_model