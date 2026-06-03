import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import sqlalchemy as sa


Model="Multilayer Perceptron"
# %%

# %%
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cpu=torch.device("cpu")
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)




class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model=MLP(input_dim=input_dim).to(device)
        self.model.apply(init_weights)

    def train(self, dataloader, optimizer):
        self.model.train()

        total_loss = 0.0      
        total_weight = 0.0

        for X, y, w in dataloader:
            X, y, w = X.to(device), y.to(device), w.to(device)

            logits = self.model(X)
            loss_per_sample = F.cross_entropy(logits, y, reduction="none")
            loss = (loss_per_sample * w).sum() / w.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += (loss_per_sample * w).sum().item()
            total_weight += w.sum().item()

        return total_loss / total_weight

    def validation(self, dataloader):
        self.model.eval()

        total_weighted_loss = 0.0
        total_weight = 0.0
        total_weight_correct = 0.0

        with torch.no_grad():
            for X, y, w in dataloader:
                X, y, w = X.to(device), y.to(device), w.to(device)

                logits = self.model(X)
                loss_per_sample = F.cross_entropy(logits, y, reduction="none")

                total_weighted_loss += (loss_per_sample * w).sum().item()
                total_weight += w.sum().item()

                pred = logits.argmax(dim=1)
                total_weight_correct += ((pred == y) * w).sum().item()

        acc = total_weight_correct / total_weight
        avg_loss = total_weighted_loss / total_weight

        return avg_loss, acc





    def fit(self, train_dataloader, val_dataloader, epochs, optimizer, scheduler=None):
                accuracay_plot = [] 

                for i in range(epochs):
                        loss=self.train( train_dataloader, optimizer)
                        val_loss, acc =self.validation(val_dataloader)
                        scheduler.step(val_loss)
                        accuracay_plot.append(acc)
                        print(f"Fold {i+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}")
                return np.max(accuracay_plot)


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = X.to(device)
            outputs = self.model(X)
            preds = outputs.argmax(dim=1)
        return preds.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X = X.to(device)
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()
            