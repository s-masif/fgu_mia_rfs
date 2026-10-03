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
class GatedOptimizedGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_clients=5, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Linear(in_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_features)
        self.dropout = dropout
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.client_gates = nn.Parameter(torch.ones(num_clients, hidden_dim) * 2.0)
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
        if client_id is not None:
            gate = torch.sigmoid(self.client_gates[client_id])
            x = x * gate
        embedding = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.matmul(adj, x)
        x = self.conv2(x)
        output = F.log_softmax(x, dim=1)
        if return_embedding:
            return output, embedding
        return output


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

def get_forget_retain_masks(clients, target_indices, num_nodes):
    """
    Build global forget/retain masks for the whole graph.
    forget_mask: True for nodes belonging to any target client.
    retain_mask: True for nodes belonging to any other client.
    """
    forget_mask = np.zeros(num_nodes, dtype=bool)
    retain_mask = np.zeros(num_nodes, dtype=bool)
    for i, client in enumerate(clients):
        nodes = client['node_indices']
        if i in target_indices:
            forget_mask[nodes] = True
        else:
            retain_mask[nodes] = True
    return forget_mask, retain_mask

# Expose the classes and main functions for import (optional but helpful)
__all__ = [
    'DATASET_CONFIGS', 'load_dataset', 'optimized_balanced_partitioning',
    'GatedOptimizedGCN', 'train_client', 'get_metrics',
    'SingleClientUnlearningEnv', 'MultiClientUnlearningEnv',
    'get_forget_retain_masks', 'set_seed'
] 


def optimized_balanced_partitioning(data_dict, num_clients, seed, louvain_resolution=1.0, louvain_delta=20):
    np.random.seed(seed)
    print(f"\n🎯 Creating {num_clients} partitions (Louvain, delta={louvain_delta})")
    G = nx.Graph()
    G.add_nodes_from(range(data_dict['num_nodes']))
    edges = data_dict['edge_index']
    if torch.is_tensor(edges):
        edges = edges.cpu().numpy()
    # Ensure edges are in (2, E) format
    if edges.ndim == 2 and edges.shape[1] == 2:
        edges = edges.T
    G.add_edges_from(edges.T)

    partition = community_louvain.best_partition(G, resolution=louvain_resolution, random_state=seed)
    groups = []
    partition_groups = {}
    for node_id, comm_id in partition.items():
        if comm_id not in groups:
            groups.append(comm_id)
            partition_groups[comm_id] = []
        partition_groups[comm_id].append(node_id)
    print(f"Found {len(groups)} natural communities")

    # Split oversized communities
    group_len_max = data_dict['num_nodes'] // num_clients - louvain_delta
    for group_i in groups[:]:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = partition_groups[group_i][:]
            partition_groups[group_i] = long_group[:group_len_max]
            new_grp_i = max(groups) + 1 if groups else 0
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    print(f"After splitting: {len(groups)} groups")

    # Bin‑packing assignment
    len_dict = {g: len(partition_groups[g]) for g in groups}
    sort_len_dict = dict(sorted(len_dict.items(), key=lambda x: x[1], reverse=True))
    owner_nodes_len = data_dict['num_nodes'] // num_clients
    owner_node_ids = {i: [] for i in range(num_clients)}
    owner_list = list(range(num_clients))
    owner_ind = 0
    give_up = 1000

    for group_i in sort_len_dict.keys():
        while len(owner_list) >= 2 and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.pop(owner_ind)
            if owner_list:
                owner_ind %= len(owner_list)
        cnt = 0
        current_owner = owner_list[owner_ind]
        while len(owner_node_ids[current_owner]) + len(partition_groups[group_i]) >= owner_nodes_len + louvain_delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
            current_owner = owner_list[owner_ind]
            cnt += 1
            if cnt > give_up:
                min_v = float('inf')
                for i, o in enumerate(owner_list):
                    if len(owner_node_ids[o]) < min_v:
                        min_v = len(owner_node_ids[o])
                        owner_ind = i
                current_owner = owner_list[owner_ind]
                break
        owner_node_ids[current_owner] += partition_groups[group_i]

    clients = [owner_node_ids[i] for i in range(num_clients)]
    # Build client data structures
    client_data_list = []
    total_assigned = 0
    for i, nodes in enumerate(clients):
        if len(nodes) == 0:
            continue
        node_indices = np.array(nodes)
        total_assigned += len(node_indices)
        global_to_local = {g: l for l, g in enumerate(node_indices)}
        # Local adjacency
        local_adj = np.zeros((len(node_indices), len(node_indices)), dtype=np.float32)
        edge_raw = data_dict['edge_index']
        if torch.is_tensor(edge_raw):
            edge_raw = edge_raw.cpu().numpy()
        if edge_raw.ndim == 2 and edge_raw.shape[0] == 2:
            edge_iter = edge_raw.T
        else:
            edge_iter = edge_raw.reshape(-1, 2)
        for src, dst in edge_iter:
            src, dst = int(src), int(dst)
            if src in global_to_local and dst in global_to_local:
                local_adj[global_to_local[src], global_to_local[dst]] = 1.0
        local_adj += np.eye(len(node_indices))
        D = np.diag(np.sum(local_adj, axis=1) ** -0.5)
        local_adj_norm = D @ local_adj @ D

        client_data_list.append({
            'id': i,
            'node_indices': node_indices,
            'features': data_dict['features'][node_indices],
            'labels': data_dict['labels'][node_indices],
            'train_mask': data_dict['train_mask'][node_indices],
            'val_mask': data_dict['val_mask'][node_indices],
            'test_mask': data_dict['test_mask'][node_indices],
            'adj': torch.FloatTensor(local_adj_norm),
            'global_to_local': global_to_local
        })
    print(f"Integrity: {total_assigned} / {data_dict['num_nodes']} nodes assigned")
    return client_data_list

# ------------------------------
#  Dataset loading – unified for all datasets
# ------------------------------
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
    # Chameleon (2,277 nodes, 5 classes) — small, Cora-scale
    "Chameleon": {"type": "wikipedia_network", "num_clients": 5, "best_accuracy": 0.65,
                  "params": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 5e-4,
                             "epochs": 5, "hid_dim": 128, "dropout": 0.3,
                             "num_rounds": 120, "momentum": 0.9, "nesterov": True},
                  "splits": [0.48, 0.32, 0.20]},
    # Squirrel (5,201 nodes, 5 classes) — WikipediaNetwork sibling
    "Squirrel": {"type": "wikipedia_network", "num_clients": 10, "best_accuracy": 0.45,
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
        dataset_name = dataset_name.lower().replace("-", "_")
        features_np, edge_index, labels_np, train_mask, val_mask, test_mask = load_heterophilous_dataset(dataset_name)
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
def train_client(model, client_data, cfg, use_momentum=True):
    model.train()
    train_idx = np.where(client_data['train_mask'])[0]
    if len(train_idx) == 0:
        return 0.0
    if cfg["params"]["optimizer"] == "sgd":
        opt = optim.SGD(model.parameters(), lr=cfg["params"]["lr"],
                        weight_decay=cfg["params"]["weight_decay"],
                        momentum=cfg["params"]["momentum"] if use_momentum else 0,
                        nesterov=cfg["params"]["nesterov"] if use_momentum else False)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg["params"]["lr"],
                         weight_decay=cfg["params"]["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.5, min_lr=1e-5)
    for _ in range(cfg["params"]["epochs"]):
        opt.zero_grad()
        out = model(client_data['features'], client_data['adj'], client_data['id'])
        loss = F.nll_loss(out[train_idx], client_data['labels'][train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        scheduler.step(loss)   # <-- scheduler step added
    return loss.item()

def get_metrics(model, clients):
    model.eval()
    local_accs = []
    all_preds, all_labels = [], []
    for i, c in enumerate(clients):
        mask = c['test_mask'] if c['test_mask'].sum() > 0 else c['val_mask']
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
class MultiClientUnlearningEnv:
    """Removes 20% of clients (used for CS, Photo, Physics, heterophilous)"""
    def __init__(self, dataset_name, seed):
        self.dataset_name = dataset_name
        self.cfg = DATASET_CONFIGS[dataset_name]
        self.data = load_dataset(dataset_name, seed)
        self.clients = optimized_balanced_partitioning(self.data, self.cfg["num_clients"], seed)
        self.model = GatedOptimizedGCN(
            in_features=self.data['num_features'],
            hidden_dim=self.cfg["params"]["hid_dim"],
            out_features=self.data['num_classes'],
            num_clients=self.cfg["num_clients"],
            dropout=self.cfg["params"]["dropout"]
        )
        self.num_target = max(1, int(0.2 * self.cfg["num_clients"]))

    def train(self):
        print(f"\n🏋️ Training {self.dataset_name} (20% removal mode)")
        best_acc = 0.0
        best_state = None
        for rnd in tqdm(range(self.cfg["params"]["num_rounds"]), desc="Rounds"):
            client_models = []
            for cl in self.clients:
                cm = copy.deepcopy(self.model)
                train_client(cm, cl, self.cfg, use_momentum=True)
                client_models.append(cm)
            # FedAvg
            new_state = self.model.state_dict()
            for key in new_state:
                stacked = torch.stack([cm.state_dict()[key].float() for cm in client_models], 0)
                new_state[key] = stacked.mean(0)
            self.model.load_state_dict(new_state)
            if (rnd+1) % 10 == 0 or rnd == self.cfg["params"]["num_rounds"]-1:
                _, acc = get_metrics(self.model, self.clients)
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(self.model.state_dict())
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"✅ Best test accuracy: {best_acc:.4f} (expected {self.cfg['best_accuracy']:.4f})")
        return best_acc

    def gated_scrub_multiple(self, target_indices):
        """Unlearn several clients (phase1: gradient ascent, phase2: lock gates, phase3: repair)"""
        # Phase 1: confuse
        unlearn_opt = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        for _ in range(5):
            total_loss = 0
            unlearn_opt.zero_grad()
            for tid in target_indices:
                c = self.clients[tid]
                mask = c['train_mask']
                if mask.sum() == 0:
                    continue
                out = self.model(c['features'], c['adj'], tid)
                loss = F.nll_loss(out[mask], c['labels'][mask])
                total_loss += loss
            if total_loss > 0:
                (-total_loss / len(target_indices)).backward()
                unlearn_opt.step()
        # Phase 2: lock gates
        with torch.no_grad():
            for tid in target_indices:
                self.model.client_gates[tid].fill_(-25.0)
        # Phase 3: repair with retained clients
        retain = [i for i in range(len(self.clients)) if i not in target_indices]
        repair_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg["params"]["lr"],
            weight_decay=self.cfg["params"]["weight_decay"],
            momentum=self.cfg["params"]["momentum"],
            nesterov=self.cfg["params"]["nesterov"]
        )
        for _ in range(5):
            for rid in retain:
                c = self.clients[rid]
                mask = c['train_mask']
                if mask.sum() == 0:
                    continue
                repair_opt.zero_grad()
                out = self.model(c['features'], c['adj'], rid)
                loss = F.nll_loss(out[mask], c['labels'][mask])
                loss.backward()
                # zero gradients for locked gates
                if self.model.client_gates.grad is not None:
                    for tid in target_indices:
                        self.model.client_gates.grad[tid].zero_()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                repair_opt.step()

    def get_target_indices(self, seed):
        np.random.seed(seed)
        all_idx = list(range(len(self.clients)))
        target = np.random.choice(all_idx, size=self.num_target, replace=False).tolist()
        return sorted(target), [i for i in all_idx if i not in target]

class SingleClientUnlearningEnv:
    """Removes one client (Planetoid datasets)"""
    def __init__(self, dataset_name, seed):
        self.dataset_name = dataset_name
        self.cfg = DATASET_CONFIGS[dataset_name]
        self.data = load_dataset(dataset_name, seed)
        self.clients = optimized_balanced_partitioning(self.data, self.cfg["num_clients"], seed)
        self.model = GatedOptimizedGCN(
            in_features=self.data['num_features'],
            hidden_dim=self.cfg["params"]["hid_dim"],
            out_features=self.data['num_classes'],
            num_clients=self.cfg["num_clients"],
            dropout=self.cfg["params"]["dropout"]
        )
        if self.cfg["target_client"] is not None:
            self.target_client = self.cfg["target_client"]
        else:
            self.target_client = random.randint(0, self.cfg["num_clients"]-1)

    def train(self):
        print(f"\n🏋️ Training {self.dataset_name} (single‑client removal mode)")
        best_acc = 0.0
        best_state = None
        for rnd in tqdm(range(self.cfg["params"]["num_rounds"]), desc="Rounds"):
            client_models = []
            for cl in self.clients:
                cm = copy.deepcopy(self.model)
                train_client(cm, cl, self.cfg, use_momentum=True)
                client_models.append(cm)
            new_state = self.model.state_dict()
            for key in new_state:
                stacked = torch.stack([cm.state_dict()[key].float() for cm in client_models], 0)
                new_state[key] = stacked.mean(0)
            self.model.load_state_dict(new_state)
            if (rnd+1) % 10 == 0 or rnd == self.cfg["params"]["num_rounds"]-1:
                _, acc = get_metrics(self.model, self.clients)
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(self.model.state_dict())
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"✅ Best test accuracy: {best_acc:.4f} (expected {self.cfg['best_accuracy']:.4f})")
        return best_acc

    def gated_scrub(self):
        # Phase 1: gradient ascent on target client
        unlearn_opt = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        c = self.clients[self.target_client]
        for _ in range(5):
            unlearn_opt.zero_grad()
            mask = c['train_mask']
            if mask.sum() > 0:
                out = self.model(c['features'], c['adj'], self.target_client)
                loss = F.nll_loss(out[mask], c['labels'][mask])
                (-loss).backward()
                unlearn_opt.step()
        # Phase 2: lock gate
        with torch.no_grad():
            self.model.client_gates[self.target_client].fill_(-25.0)
        # Phase 3: repair with other clients
        retain = [i for i in range(len(self.clients)) if i != self.target_client]
        repair_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg["params"]["lr"],
            weight_decay=self.cfg["params"]["weight_decay"],
            momentum=self.cfg["params"]["momentum"],
            nesterov=self.cfg["params"]["nesterov"]
        )
        for _ in range(5):
            for rid in retain:
                c = self.clients[rid]
                mask = c['train_mask']
                if mask.sum() == 0:
                    continue
                repair_opt.zero_grad()
                out = self.model(c['features'], c['adj'], rid)
                loss = F.nll_loss(out[mask], c['labels'][mask])
                loss.backward()
                if self.model.client_gates.grad is not None:
                    self.model.client_gates.grad[self.target_client].zero_()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                repair_opt.step()

# ------------------------------
#  Experiment runner (multi‑seed aggregation)
# ------------------------------
def run_all_experiments():
    # All datasets from the three original files
    all_datasets = ["Cora", "CiteSeer", "PubMed", "CS", "Photo", "Physics",
                    "tolokers", "minesweeper", "amazon_ratings"]
    seeds = [2024, 2025, 2026]
    results_dir = "unified_unlearning_results"
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []

    for ds in all_datasets:
        print("\n" + "="*80)
        print(f"🚀 {ds} – running 3 seeds")
        print("="*80)
        cfg = DATASET_CONFIGS[ds]
        pre_accs, post_accs, forgetting_ratios, retention_ratios = [], [], [], []
        target_drops = []

        for seed in seeds:
            set_seed(seed)
            if cfg["type"] == "planetoid":
                env = SingleClientUnlearningEnv(ds, seed)
                env.train()
                pre_local, pre_global = get_metrics(env.model, env.clients)
                target_idx = env.target_client
                env.gated_scrub()
                post_local, post_global = get_metrics(env.model, env.clients)
                target_pre = pre_local[target_idx]
                target_post = post_local[target_idx]
                retain_pre = np.mean([pre_local[i] for i in range(len(pre_local)) if i != target_idx])
                retain_post = np.mean([post_local[i] for i in range(len(post_local)) if i != target_idx])
            else:
                env = MultiClientUnlearningEnv(ds, seed)
                env.train()
                pre_local, pre_global = get_metrics(env.model, env.clients)
                target_idx, _ = env.get_target_indices(seed)
                env.gated_scrub_multiple(target_idx)
                post_local, post_global = get_metrics(env.model, env.clients)
                target_pre = np.mean([pre_local[i] for i in target_idx])
                target_post = np.mean([post_local[i] for i in target_idx])
                retain_idx = [i for i in range(len(pre_local)) if i not in target_idx]
                retain_pre = np.mean([pre_local[i] for i in retain_idx])
                retain_post = np.mean([post_local[i] for i in retain_idx])

            forgetting = (target_pre - target_post) / max(target_pre, 1e-6)
            retention = retain_post / max(retain_pre, 1e-6)

            pre_accs.append(pre_global)
            post_accs.append(post_global)
            forgetting_ratios.append(forgetting)
            retention_ratios.append(retention)
            target_drops.append(target_pre - target_post)

            print(f"  Seed {seed}: pre={pre_global:.4f} post={post_global:.4f} "
                  f"forgetting={forgetting:.3f} retention={retention:.3f}")

        summary_rows.append({
            "Dataset": ds,
            "Pre (mean±std)": f"{np.mean(pre_accs):.4f}±{np.std(pre_accs):.4f}",
            "Post (mean±std)": f"{np.mean(post_accs):.4f}±{np.std(post_accs):.4f}",
            "Forgetting ratio": f"{np.mean(forgetting_ratios):.4f}±{np.std(forgetting_ratios):.4f}",
            "Retention ratio": f"{np.mean(retention_ratios):.4f}±{np.std(retention_ratios):.4f}",
            "Target accuracy drop": f"{np.mean(target_drops):.4f}±{np.std(target_drops):.4f}"
        })

    # Save results
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(results_dir, "unified_summary.csv"), index=False)
    print("\n" + "="*80)
    print("FINAL SUMMARY (mean ± std over 3 seeds)")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nResults saved to {results_dir}/unified_summary.csv")

if __name__ == "__main__":
    run_all_experiments()