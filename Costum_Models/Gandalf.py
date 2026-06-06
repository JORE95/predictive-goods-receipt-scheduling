
import torch
from torch import Tensor
from typing import Callable
import torch.nn as nn
import torch
import random
import numpy as np  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   



def t_softmax(input: Tensor, t: Tensor = None, dim: int = -1) -> Tensor:
    if t is None:
        t = torch.tensor(0.5, device=input.device)
    assert (t >= 0.0).all()
    maxes = torch.max(input, dim=dim, keepdim=True).values
    input_minus_maxes = input - maxes

    w = torch.relu(input_minus_maxes + t) + 1e-8
    return torch.softmax(input_minus_maxes + torch.log(w), dim=dim)


class TSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, t: Tensor) -> Tensor:
        return t_softmax(input, t, self.dim)


class RSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.tsoftmax = TSoftmax(dim=dim)

    @classmethod
    def calculate_t(cls, input: Tensor, r: Tensor, dim: int = -1, eps: float = 1e-8):
        assert ((0.0 <= r) & (r <= 1.0)).all()

        maxes = torch.max(input, dim=dim, keepdim=True).values
        input_minus_maxes = input - maxes

        zeros_mask = torch.exp(input_minus_maxes) == 0.0
        zeros_frac = zeros_mask.sum(dim=dim, keepdim=True).float() / input_minus_maxes.shape[dim]

        q = torch.clamp((r - zeros_frac) / (1 - zeros_frac), min=0.0, max=1.0)
        x_minus_maxes = input_minus_maxes * (~zeros_mask).float()
        if q.ndim > 1:
            t = -torch.quantile(x_minus_maxes, q.view(-1), dim=dim, keepdim=True).detach()
            t = t.squeeze(dim).diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + eps
        else:
            t = -torch.quantile(x_minus_maxes, q, dim=dim).detach() + eps
        return t

    def forward(self, input: Tensor, r: Tensor):
        t = RSoftmax.calculate_t(input, r, self.dim, self.eps)
        return self.tsoftmax(input, t)


class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_stages: int,
        feature_mask_function: Callable = t_softmax,
        feature_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self.feature_mask_function = feature_mask_function
        self._dropout = dropout
        self.n_stages = n_stages
        self.feature_sparsity = feature_sparsity
        self.learnable_sparsity = learnable_sparsity
        self._build_network()


    def _create_feature_mask(self):
        feature_masks = torch.cat(
            [
                torch.distributions.Beta(
                    torch.tensor([random.uniform(0.5, 10.0)]),
                    torch.tensor([random.uniform(0.5, 10.0)]),
                )
                .sample((self.n_features_in,))
                .squeeze(-1)
                for _ in range(self.n_stages)
            ]
        ).reshape(self.n_stages, self.n_features_in)
        return nn.Parameter(
            feature_masks,
            requires_grad=True,
        )

    def _build_network(self):
        self.W_in = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, 2 * self.n_features_in) for _ in range(self.n_stages)]
        )
        self.W_out = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, self.n_features_in) for _ in range(self.n_stages)]
        )

        self.feature_masks = self._create_feature_mask()
        if self.feature_mask_function.__name__ == "t_softmax":
            t = RSoftmax.calculate_t(self.feature_masks, r=torch.tensor([self.feature_sparsity]), dim=-1)
            self.t = nn.Parameter(t, requires_grad=self.learnable_sparsity)
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        t = torch.relu(self.t) if self.feature_mask_function.__name__ == "t_softmax" else None
        for d in range(self.n_stages):
            if self.feature_mask_function.__name__ == "t_softmax":
                feature = self.feature_mask_function(self.feature_masks[d], t[d]) * x
            else:
                feature = self.feature_mask_function(self.feature_masks[d]) * h
            h_in = self.W_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])  # noqa: E203
            h_out = torch.tanh(self.W_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h


class GANDALFBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        gflu_stages: int,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
   
    ):
        super().__init__()
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.input_dim = input_dim
        self._build_network()

    def _build_network(self):
        self.gflus = GatedFeatureLearningUnit(
            n_features_in=self.input_dim,
            n_stages=self.gflu_stages,
            feature_mask_function=t_softmax,
            dropout=self.gflu_dropout,
            feature_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
        )

        self.mlps =  nn.Sequential(
            nn.Linear(self.input_dim,128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.gflus(x)
        return self.mlps(x)

# %%
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



# %%

class Model(nn.Module):
    def __init__(self, input_dim, gflu_stages, gflu_dropout, gflu_feature_init_sparsity, learnable_sparsity):
        super().__init__()
        self.model = GANDALFBackbone(
            input_dim=input_dim,
            gflu_stages=gflu_stages,
            gflu_dropout=gflu_dropout,
            gflu_feature_init_sparsity=gflu_feature_init_sparsity,
            learnable_sparsity=learnable_sparsity,
        ).to(device)

        self.model.apply(init_weights)


    def train(self, dataloader, optimizer, loss_fn):
        self.model.train()

        total_weighted_loss = 0.0
        total_weight = 0.0

        for X_batch, y_batch, w_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)

            optimizer.zero_grad()

            outputs = self.model(X_batch)

            loss_per_sample = loss_fn(outputs, y_batch)
            weighted_loss = (loss_per_sample * w_batch).sum() / w_batch.sum()
            weighted_loss.backward()
            optimizer.step()

            total_weighted_loss += (loss_per_sample * w_batch).sum().item()
            total_weight += w_batch.sum().item()

        return total_weighted_loss / total_weight



    def validation(self, dataloader, loss_fn):
        self.model.eval()

        total_weighted_loss = 0.0
        total_weight = 0.0
        correct = 0.0

        with torch.no_grad():
            for X_batch, y_batch, w_batch in dataloader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                w_batch = w_batch.to(device)

                outputs = self.model(X_batch)

                loss_per_sample = loss_fn(outputs, y_batch)
                total_weighted_loss += (loss_per_sample * w_batch).sum().item()
                total_weight += w_batch.sum().item()

                preds = outputs.argmax(dim=1)

                correct += ((preds == y_batch) * w_batch).sum().item()

        avg_loss = total_weighted_loss / total_weight
        accuracy = correct / total_weight

        return avg_loss, accuracy
    
    def fit(self, train_dataloader, val_dataloader, epochs, optimizer, sheduler, loss_fn):
                accuracay_plot = [] 

                for i in range(epochs):
                        loss=self.train(train_dataloader, optimizer,loss_fn)
                        val_loss, acc =self.validation(val_dataloader,loss_fn)
                        sheduler.step(val_loss)
                        accuracay_plot.append(acc)
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