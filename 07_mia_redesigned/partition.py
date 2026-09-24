"""Graph partitioning, client construction, and per-client train/val/test split.

Each client dict carries `node_indices`: the ORIGINAL-graph index of every local
node. This is the stable global node identity used for the per-node logs and for
enforcing identity-disjoint attack pools (see core/attack.py and core/logging.py).
"""
import numpy as np
import torch
import networkx as nx
import community as community_louvain

from models import DATASET_SPLIT_RATIOS

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



def apply_per_client_split(clients: list, dataset_name: str,
                           seed: int = 0) -> list:
    """Apply per-client train/val/test split with dataset-specific ratios.

    Partitioning has already happened; this re-assigns each client's
    train_mask, val_mask, test_mask using a random permutation of its OWN
    local nodes at the ratio specified for that dataset.
    """
    train_r, val_r, _ = DATASET_SPLIT_RATIOS.get(dataset_name, (0.20, 0.40, 0.40))
    rng = np.random.default_rng(seed)
    for cl in clients:
        n    = cl["features"].shape[0]
        perm = rng.permutation(n)
        t1   = int(n * train_r)
        t2   = int(n * (train_r + val_r))
        train_mask = np.zeros(n, dtype=bool); train_mask[perm[:t1]]   = True
        val_mask   = np.zeros(n, dtype=bool); val_mask[perm[t1:t2]]   = True
        test_mask  = np.zeros(n, dtype=bool); test_mask[perm[t2:]]    = True
        cl["train_mask"], cl["val_mask"], cl["test_mask"] = train_mask, val_mask, test_mask
    return clients