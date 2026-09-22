"""Core models, dataset loaders, and configuration (unchanged machinery)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import os
import pandas as pd
from tqdm import tqdm
import warnings
import random
import requests
import io
import json

warnings.filterwarnings('ignore')

# ------------------------------
#  Global setup & helpers
# ------------------------------

# ── reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ------------------------------
#  Gated GCN model (identical for all cases)
# ------------------------------

# ── model ────────────────────────────────────────────────────────────────────
class PlainGCN(nn.Module):
    """Plain 2-layer GCN — identical to GatedOptimizedGCN minus the per-client
    gate. Used as the retrain-from-scratch baseline for MIA-consistency
    experiments where we want a standard architecture with no custom
    personalization mechanism.

    Accepts a `client_id` argument for API compatibility with `train_client`
    and `get_metrics`, but ignores it (no gates to index into).
    """
    def __init__(self, in_features, hidden_dim, out_features, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Linear(in_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_features)
        self.dropout = dropout
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, adj, client_id=None, return_embedding=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.matmul(adj, x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        embedding = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.matmul(adj, x)
        x = self.conv2(x)
        output = F.log_softmax(x, dim=1)
        if return_embedding:
            return output, embedding
        return output

# ------------------------------
#  Robust graph partitioning (works for all edge formats)
# ------------------------------
    
# ========== ADDITIONS FOR VERIFICATION EXPERIMENTS ==========


# ── per-dataset configuration ────────────────────────────────────────────────
DATASET_CONFIGS = {
    # Planetoid
    "Cora": {"type": "planetoid", "num_clients": 5, "best_accuracy": 0.87,
             "params": {"optimizer": "sgd", "lr": 0.005, "weight_decay": 5e-4,
                        "epochs": 4, "hid_dim": 256, "dropout": 0.5,
                        "num_rounds": 120, "momentum": 0.9, "nesterov": True},
             "target_client": 1},
    "CiteSeer": {"type": "planetoid", "num_clients": 5, "best_accuracy": 0.699,
                 "params": {"optimizer": "sgd", "lr": 0.005, "weight_decay": 5e-4,
                            "epochs": 3, "hid_dim": 128, "dropout": 0.5,
                            "num_rounds": 120, "momentum": 0.9, "nesterov": True},
                 "target_client": None},
    "PubMed": {"type": "planetoid", "num_clients": 5, "best_accuracy": 0.87,
               "params": {"optimizer": "adam", "lr": 0.01, "weight_decay": 5e-4,
                          "epochs": 2, "hid_dim": 128, "dropout": 0.5,
                          "num_rounds": 100, "momentum": 0.9, "nesterov": False},
               "target_client": 0},
    # Coauthor / Amazon
    "CS": {"type": "coauthor", "num_clients": 10, "best_accuracy": 0.9217,
           "params": {"optimizer": "sgd", "lr": 0.001, "weight_decay": 0.0,
                      "epochs": 5, "hid_dim": 128, "dropout": 0.5,
                      "num_rounds": 120, "momentum": 0.9, "nesterov": True},
           "splits": [0.2, 0.4, 0.4]},
    "Photo": {"type": "amazon", "num_clients": 10, "best_accuracy": 0.9013,
              "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0,
                         "epochs": 5, "hid_dim": 128, "dropout": 0.5,
                         "num_rounds": 120, "momentum": 0.9, "nesterov": True},
              "splits": [0.2, 0.4, 0.4]},
    "Physics": {"type": "coauthor", "num_clients": 15, "best_accuracy": 0.9536,
                "params": {"optimizer": "sgd", "lr": 0.001, "weight_decay": 0.0,
                           "epochs": 5, "hid_dim": 128, "dropout": 0.3,
                           "num_rounds": 120, "momentum": 0.9, "nesterov": True},
                "splits": [0.2, 0.4, 0.4]},
    # Heterophilous
    "tolokers": {"type": "heterophilous", "num_clients": 15, "best_accuracy": 0.7891,
                 "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0005,
                            "epochs": 5, "hid_dim": 128, "dropout": 0.5,
                            "num_rounds": 120, "momentum": 0.9, "nesterov": True}},
    "minesweeper": {"type": "heterophilous", "num_clients": 15, "best_accuracy": 0.8072,
                    "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0,
                               "epochs": 3, "hid_dim": 64, "dropout": 0.3,
                               "num_rounds": 120, "momentum": 0.9, "nesterov": True}},
    "amazon_ratings": {"type": "heterophilous", "num_clients": 20, "best_accuracy": 0.4197,
                       "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0,
                                  "epochs": 5, "hid_dim": 128, "dropout": 0.3,
                                  "num_rounds": 120, "momentum": 0.9, "nesterov": True}},

    # ── NEW: additional homophilic datasets ──────────────────────────────────
    # Amazon Computers (13,381 nodes, 10 classes) — same loader as Photo
    "Computers": {"type": "amazon", "num_clients": 10, "best_accuracy": 0.85,
                  "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0,
                             "epochs": 5, "hid_dim": 128, "dropout": 0.5,
                             "num_rounds": 120, "momentum": 0.9, "nesterov": True},
                  "splits": [0.2, 0.4, 0.4]},
    # WikiCS (11,701 nodes, 10 classes) — Wikipedia CS-article graph
    "WikiCS": {"type": "wiki_cs", "num_clients": 10, "best_accuracy": 0.80,
               "params": {"optimizer": "adam", "lr": 0.005, "weight_decay": 5e-4,
                          "epochs": 5, "hid_dim": 128, "dropout": 0.5,
                          "num_rounds": 120, "momentum": 0.9, "nesterov": False},
               "splits": [0.2, 0.4, 0.4]},

    # ── NEW: additional heterophilic datasets ────────────────────────────────
    # Chameleon FILTERED (Platonov et al. 2023, ~890 nodes) — duplicates removed
    "Chameleon": {"type": "heterophilous", "npz_name": "chameleon_filtered",
                  "num_clients": 5, "best_accuracy": 0.65,
                  "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 5e-4,
                             "epochs": 5, "hid_dim": 128, "dropout": 0.3,
                             "num_rounds": 120, "momentum": 0.9, "nesterov": True},
                  "splits": [0.48, 0.32, 0.20]},
    # Squirrel FILTERED (Platonov et al. 2023, ~2,223 nodes) — duplicates removed
    "Squirrel": {"type": "heterophilous", "npz_name": "squirrel_filtered",
                 "num_clients": 10, "best_accuracy": 0.45,
                 "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 5e-4,
                            "epochs": 5, "hid_dim": 128, "dropout": 0.3,
                            "num_rounds": 120, "momentum": 0.9, "nesterov": True},
                 "splits": [0.48, 0.32, 0.20]},
    # Actor (7,600 nodes, 5 classes) — actor co-occurrence, strongly heterophilic
    "Actor": {"type": "actor", "num_clients": 10, "best_accuracy": 0.35,
              "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 5e-4,
                         "epochs": 5, "hid_dim": 128, "dropout": 0.3,
                         "num_rounds": 120, "momentum": 0.9, "nesterov": True},
              "splits": [0.5, 0.25, 0.25]},
    # Roman-empire (22,662 nodes, 18 classes) — uses existing Yandex loader
    "roman_empire": {"type": "heterophilous", "num_clients": 15, "best_accuracy": 0.70,
                     "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 5e-4,
                                "epochs": 5, "hid_dim": 128, "dropout": 0.3,
                                "num_rounds": 120, "momentum": 0.9, "nesterov": True}},
}

DATASET_SPLIT_RATIOS = {
    "Cora":           (0.20, 0.40, 0.40),
    "PubMed":         (0.20, 0.40, 0.40),
    "CS":             (0.20, 0.40, 0.40),
    "Photo":          (0.20, 0.40, 0.40),
    "tolokers":       (0.50, 0.25, 0.25),
    "minesweeper":    (0.50, 0.25, 0.25),
    "amazon_ratings": (0.50, 0.25, 0.25),
    # ── NEW ──────────────────────────────────────────────────────────────
    "Computers":      (0.20, 0.40, 0.40),   # matches Photo
    "WikiCS":         (0.20, 0.40, 0.40),   # standard homophilic
    "Chameleon":      (0.48, 0.32, 0.20),   # WikipediaNetwork convention
    "Squirrel":       (0.48, 0.32, 0.20),   # WikipediaNetwork convention
    "Actor":          (0.50, 0.25, 0.25),   # heterophilic convention
    "roman_empire":   (0.50, 0.25, 0.25),   # heterophilic (Yandex)
}

# ── dataset loaders ──────────────────────────────────────────────────────────
def load_heterophilous_dataset(name):
    url_base = "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/"
    fname = f"{name}.npz"
    resp = requests.get(url_base + fname)
    resp.raise_for_status()
    data = np.load(io.BytesIO(resp.content), allow_pickle=True)
    node_features = data['node_features']
    edges = data['edges']
    labels = data['node_labels']
    # Use first split from masks
    train_mask = data['train_masks'][0]
    val_mask = data['val_masks'][0]
    test_mask = data['test_masks'][0]
    return node_features, edges, labels, train_mask, val_mask, test_mask


def load_dataset(dataset_name, seed):
    print(f"\n📥 Loading {dataset_name}")
    cfg = DATASET_CONFIGS[dataset_name]
    typ = cfg["type"]
    np.random.seed(seed)

    if typ == "planetoid":
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
        data = dataset[0]
        features = data.x
        labels = data.y
        edge_index = data.edge_index.numpy()
        if dataset_name == "PubMed":
            # 20/40/40 split
            indices = np.random.permutation(data.num_nodes)
            train_mask = np.zeros(data.num_nodes, dtype=bool)
            val_mask = np.zeros(data.num_nodes, dtype=bool)
            test_mask = np.zeros(data.num_nodes, dtype=bool)
            train_mask[indices[:int(0.2*data.num_nodes)]] = True
            val_mask[indices[int(0.2*data.num_nodes):int(0.6*data.num_nodes)]] = True
            test_mask[indices[int(0.6*data.num_nodes):]] = True
        else:
            train_mask = data.train_mask.numpy()
            val_mask = data.val_mask.numpy()
            test_mask = data.test_mask.numpy()
        num_classes = dataset.num_classes
        num_features = features.shape[1]
        num_nodes = data.num_nodes

    elif typ in ("coauthor", "amazon", "wikipedia_network", "actor", "wiki_cs"):
        if typ == "coauthor":
            dataset = Coauthor(root='/tmp', name=dataset_name)
        elif typ == "amazon":
            dataset = Amazon(root='/tmp', name=dataset_name)
        elif typ == "wikipedia_network":
            from torch_geometric.datasets import WikipediaNetwork
            dataset = WikipediaNetwork(root=f'/tmp/{dataset_name}',
                                        name=dataset_name.lower(),
                                        geom_gcn_preprocess=True)
        elif typ == "actor":
            from torch_geometric.datasets import Actor
            dataset = Actor(root='/tmp/Actor')
        elif typ == "wiki_cs":
            from torch_geometric.datasets import WikiCS
            dataset = WikiCS(root='/tmp/WikiCS')
        data = dataset[0]
        features = data.x
        labels = data.y
        edge_index = data.edge_index.numpy()
        splits = cfg["splits"]
        indices = np.random.permutation(data.num_nodes)
        train_end = int(splits[0] * data.num_nodes)
        val_end = int((splits[0]+splits[1]) * data.num_nodes)
        train_mask = np.zeros(data.num_nodes, dtype=bool)
        val_mask = np.zeros(data.num_nodes, dtype=bool)
        test_mask = np.zeros(data.num_nodes, dtype=bool)
        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True
        num_classes = dataset.num_classes
        num_features = features.shape[1]
        num_nodes = data.num_nodes

    elif typ == "heterophilous":
        # Honor an explicit npz_name if the config provides one (e.g. the
        # Platonov filtered Chameleon/Squirrel), else fall back to the
        # lower-cased dataset name.
        npz_name = cfg.get("npz_name", dataset_name.lower().replace("-", "_"))
        features_np, edge_index, labels_np, train_mask, val_mask, test_mask = load_heterophilous_dataset(npz_name)
        # Yandex .npz stores edges as (E, 2); the adjacency build below expects
        # (2, E). Transpose if needed so edge_index[0]/[1] are src/dst arrays.
        edge_index = np.asarray(edge_index)
        if edge_index.ndim == 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.T
        features = torch.FloatTensor(features_np)
        labels = torch.LongTensor(labels_np)
        num_nodes = features.shape[0]
        num_features = features.shape[1]
        num_classes = len(np.unique(labels_np))

    else:
        raise ValueError(f"Unknown dataset type {typ}")

    # Build normalized adjacency
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj += np.eye(num_nodes)
    D = np.diag(np.sum(adj, axis=1) ** -0.5)
    adj_norm = D @ adj @ D

    print(f"  Nodes: {num_nodes}, Features: {num_features}, Classes: {num_classes}")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    return {
        'name': dataset_name,
        'features': features,
        'labels': labels,
        'adj': torch.FloatTensor(adj_norm),
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes,
        'edge_index': edge_index,
        'config': cfg
    }

# ------------------------------
#  Training helpers (with scheduler)
# ------------------------------

# ── evaluation helper (val for selection, test only for final reporting) ─────
def get_metrics(model, clients, mask_type='val'):
    """Evaluate accuracy on the chosen mask.
    mask_type='val'  -> model selection (no test leakage)
    mask_type='test' -> final reporting only
    """
    model.eval()
    local_accs = []
    all_preds, all_labels = [], []
    for i, c in enumerate(clients):
        if mask_type == 'val':
            # Validation-only selection. A client with no validation nodes is
            # skipped for this metric; the test set is never used here (Point 2).
            mask = c['val_mask']
            if mask.sum() == 0:
                continue
        else:
            mask = c['test_mask']
            if mask.sum() == 0:
                continue
        with torch.no_grad():
            out = model(c['features'], c['adj'], i)
            preds = out[mask].argmax(1)
            true = c['labels'][mask].cpu().numpy()
            pred = preds.cpu().numpy()
            acc = accuracy_score(true, pred) if len(true) > 0 else 0.0
            local_accs.append(acc)
            all_preds.extend(pred)
            all_labels.extend(true)
    global_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return local_accs, global_acc

# ------------------------------
#  Unlearning environments
# ------------------------------