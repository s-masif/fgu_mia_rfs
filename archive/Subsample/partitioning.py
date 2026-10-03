"""
partitioning.py
===============
Partitioning dispatcher for the variations sweep. Exposes one entry point:

    partition_graph(data_dict, num_clients, method, seed)

where `method ∈ {"louvain", "metis", "metis_plus", "random"}`.

All partitioners produce client dicts with the SAME schema as the pipeline's
existing `optimized_balanced_partitioning`:

    {
      "id"              : int,
      "node_indices"    : np.ndarray  — global node ids in this client,
      "features"        : Tensor      — X[node_indices],
      "labels"          : Tensor      — y[node_indices],
      "train_mask"      : np.ndarray  — bool, local (per-client) mask,
      "val_mask"        : np.ndarray  — same,
      "test_mask"       : np.ndarray  — same,
      "adj"             : Tensor      — D^(-1/2) A D^(-1/2), dense,
      "global_to_local" : dict        — {global_id -> local_id}
    }

so it drops in wherever `optimized_balanced_partitioning` was used.

The Metis / Metis+ logic is lifted from ef_ul_run.py but rewrapped to
produce the schema above (rather than ef_ul_run.py's edge_index-based one).
"""
from __future__ import annotations

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helper — build one client dict from a list of global node indices
# ═══════════════════════════════════════════════════════════════════════════

def _build_client_dict(client_id: int, node_indices, data_dict: dict) -> dict:
    """Assemble one client's dict from its assigned global node indices.

    Duplicates the exact logic used by `optimized_balanced_partitioning` so
    downstream code sees an indistinguishable output.
    """
    node_indices = np.asarray(node_indices, dtype=np.int64)
    global_to_local = {int(g): l for l, g in enumerate(node_indices)}

    # Local adjacency
    local_adj = np.zeros((len(node_indices), len(node_indices)), dtype=np.float32)
    edge_raw = data_dict["edge_index"]
    if torch.is_tensor(edge_raw):
        edge_raw = edge_raw.cpu().numpy()
    edge_iter = edge_raw.T if (edge_raw.ndim == 2 and edge_raw.shape[0] == 2) else edge_raw.reshape(-1, 2)

    for src, dst in edge_iter:
        src, dst = int(src), int(dst)
        if src in global_to_local and dst in global_to_local:
            local_adj[global_to_local[src], global_to_local[dst]] = 1.0

    local_adj += np.eye(len(node_indices))
    deg = np.sum(local_adj, axis=1)
    deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    D = np.diag(deg_inv_sqrt)
    local_adj_norm = D @ local_adj @ D

    return {
        "id":              client_id,
        "node_indices":    node_indices,
        "features":        data_dict["features"][node_indices],
        "labels":          data_dict["labels"][node_indices],
        "train_mask":      data_dict["train_mask"][node_indices],
        "val_mask":        data_dict["val_mask"][node_indices],
        "test_mask":       data_dict["test_mask"][node_indices],
        "adj":             torch.FloatTensor(local_adj_norm),
        "global_to_local": global_to_local,
    }


def _clients_from_membership(membership: np.ndarray, num_clients: int,
                              data_dict: dict) -> list:
    """Given a (n_nodes,) array assigning each node to a client, build
    all client dicts."""
    clients = []
    for cid in range(num_clients):
        nodes = np.where(membership == cid)[0]
        if len(nodes) == 0:
            continue
        clients.append(_build_client_dict(cid, nodes, data_dict))
    return clients


def _sanitize_edges_for_metis(edge_index: np.ndarray, n: int
                               ) -> np.ndarray:
    """Clean up edges before handing them to pymetis, which segfaults on:
      - self-loops
      - duplicate edges
      - asymmetric edges (missing reverse)
      - isolated nodes (degree 0)

    Returns a (2, m) symmetric, deduplicated, self-loop-free edge_index.
    Isolated nodes are patched by adding a self-loop back — Metis tolerates
    one, and it prevents the "degree-0 node" crash.
    """
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()

    src, dst = edge_index[0], edge_index[1]

    # 1. Remove self-loops
    keep = src != dst
    src, dst = src[keep], dst[keep]

    # 2. Symmetrize + dedupe (Metis wants each undirected edge listed both ways)
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    undirected = np.unique(np.stack([a, b]), axis=1)
    src = np.concatenate([undirected[0], undirected[1]])
    dst = np.concatenate([undirected[1], undirected[0]])

    # 3. Detect isolated nodes and give them a self-loop
    seen = np.zeros(n, dtype=bool)
    seen[src] = True
    isolated = np.where(~seen)[0]
    if len(isolated) > 0:
        print(f"   ⚠ {len(isolated)} isolated nodes — patching with self-loops")
        src = np.concatenate([src, isolated])
        dst = np.concatenate([dst, isolated])

    return np.stack([src, dst])


# ═══════════════════════════════════════════════════════════════════════════
#  Partitioner 1 — Louvain  (existing pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def partition_louvain(data_dict: dict, num_clients: int, seed: int,
                       **kwargs) -> list:
    from all_client_unlearning import optimized_balanced_partitioning
    return optimized_balanced_partitioning(data_dict, num_clients, seed)


# ═══════════════════════════════════════════════════════════════════════════
#  Partitioner 2 — Metis  (lifted from ef_ul_run.py:metis_partition)
# ═══════════════════════════════════════════════════════════════════════════

def partition_metis(data_dict: dict, num_clients: int, seed: int,
                     **kwargs) -> list:
    """Standard Metis partitioning via pymetis. Requires `pip install pymetis`."""
    try:
        import pymetis
    except ImportError:
        raise ImportError(
            "pymetis not found. Install with: pip install pymetis\n"
            "(or: conda install -c conda-forge pymetis)"
        )

    n  = data_dict["num_nodes"]
    ei = data_dict["edge_index"]

    # Sanitize edges — pymetis segfaults on self-loops, dupes, isolated nodes
    ei_clean = _sanitize_edges_for_metis(ei, n)

    # Build adjacency list (pymetis format)
    adj = [[] for _ in range(n)]
    for i in range(ei_clean.shape[1]):
        adj[int(ei_clean[0, i])].append(int(ei_clean[1, i]))

    print(f"🎯 Metis partitioning → {num_clients} clients")
    n_cuts, membership = pymetis.part_graph(num_clients, adjacency=adj)
    print(f"   Metis cuts={n_cuts}")

    return _clients_from_membership(np.asarray(membership), num_clients, data_dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Partitioner 3 — Metis+  (lifted from ef_ul_run.py:metis_plus_partition)
# ═══════════════════════════════════════════════════════════════════════════

def partition_metis_plus(data_dict: dict, num_clients: int, seed: int,
                          metis_num_coms: int = 100, **kwargs) -> list:
    """Metis-Plus (OpenFGL): Metis into `metis_num_coms` fine-grained
    communities, then KMeans on label distributions to merge into num_clients
    label-skewed clients. Simulates non-IID federated heterogeneity.
    """
    try:
        import pymetis
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(f"Missing dependency for Metis+: {e}")

    n = data_dict["num_nodes"]
    ei = data_dict["edge_index"]

    # Sanitize edges — pymetis segfaults on self-loops, dupes, isolated nodes
    ei_clean = _sanitize_edges_for_metis(ei, n)

    y = data_dict["labels"]
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    num_classes = int(y.max()) + 1

    adj = [[] for _ in range(n)]
    for i in range(ei_clean.shape[1]):
        adj[int(ei_clean[0, i])].append(int(ei_clean[1, i]))

    print(f"🎯 Metis-Plus: Metis({metis_num_coms} communities) → KMeans → {num_clients} clients")
    _, fine_membership = pymetis.part_graph(metis_num_coms, adjacency=adj)
    fine_membership = np.asarray(fine_membership)

    # Label distribution of each fine-grained community
    label_dist = np.zeros((metis_num_coms, num_classes), dtype=np.float32)
    for node_id, com_id in enumerate(fine_membership):
        label_dist[com_id, int(y[node_id])] += 1.0
    row_sums = label_dist.sum(axis=1, keepdims=True)
    label_dist = np.divide(label_dist, row_sums, where=row_sums > 0)

    # KMeans to merge fine communities → num_clients coarse clients
    km = KMeans(n_clusters=num_clients, random_state=seed, n_init=10)
    km.fit(label_dist)
    fine_to_coarse = km.labels_   # (metis_num_coms,)

    # Coarse membership per node
    membership = np.array([fine_to_coarse[fm] for fm in fine_membership])
    return _clients_from_membership(membership, num_clients, data_dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Partitioner 4 — Random
# ═══════════════════════════════════════════════════════════════════════════

def partition_random(data_dict: dict, num_clients: int, seed: int,
                      **kwargs) -> list:
    """Random uniform node partition. Baseline for the partitioning sweep."""
    n   = data_dict["num_nodes"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    membership = np.zeros(n, dtype=np.int64)
    chunks = np.array_split(perm, num_clients)
    for cid, nodes in enumerate(chunks):
        membership[nodes] = cid
    print(f"🎯 Random partitioning → {num_clients} clients "
          f"(sizes: {[len(c) for c in chunks]})")
    return _clients_from_membership(membership, num_clients, data_dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

_PARTITIONERS = {
    "louvain":    partition_louvain,
    "metis":      partition_metis,
    "metis_plus": partition_metis_plus,
    "random":     partition_random,
}


def partition_graph(data_dict: dict, num_clients: int,
                     method: str, seed: int, **kwargs) -> list:
    """Route to the requested partitioner.

    Parameters
    ----------
    data_dict    : dict from `load_dataset` (has features, edge_index, labels, masks, num_nodes)
    num_clients  : K
    method       : one of 'louvain', 'metis', 'metis_plus', 'random'
    seed         : random seed (used by the partitioner)
    **kwargs     : forwarded (e.g. metis_num_coms for metis_plus)
    """
    if method not in _PARTITIONERS:
        raise ValueError(
            f"Unknown partitioning method: {method!r}. "
            f"Choose from {list(_PARTITIONERS)}"
        )
    return _PARTITIONERS[method](data_dict, num_clients, seed, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  Modularity  (a partition-quality metric used later in correlation analysis)
# ═══════════════════════════════════════════════════════════════════════════

def partition_modularity(clients: list, data_dict: dict) -> float:
    """Newman-Girvan modularity Q of the partition.

    Q = sum_c [ (m_c / m) - (d_c / 2m)^2 ]

    where m_c = edges inside client c, d_c = total degree of client c's nodes,
    m = total edges in the graph.

    Q > 0  → partition better than random; higher = tighter communities.
    Q ~ 0  → partition ≈ random.
    """
    ei = data_dict["edge_index"]
    if torch.is_tensor(ei):
        ei = ei.cpu().numpy()
    # Undirected: treat each edge once
    if ei.ndim == 2 and ei.shape[0] == 2:
        src_all, dst_all = ei[0], ei[1]
    else:
        src_all, dst_all = ei[:, 0], ei[:, 1]

    n = data_dict["num_nodes"]
    node_to_cid = np.full(n, -1, dtype=np.int64)
    for cl in clients:
        node_to_cid[cl["node_indices"]] = int(cl["id"])

    degree = np.zeros(n, dtype=np.float64)
    for s, d in zip(src_all, dst_all):
        degree[int(s)] += 1
    total_m = float(degree.sum()) / 2.0
    if total_m == 0:
        return 0.0

    # Edges inside each community
    intra = {}
    for s, d in zip(src_all, dst_all):
        cs, cd = node_to_cid[int(s)], node_to_cid[int(d)]
        if cs == cd and cs >= 0:
            intra[cs] = intra.get(cs, 0.0) + 0.5   # undirected

    # Total degree per community
    comm_deg = {}
    for node_id in range(n):
        cid = node_to_cid[node_id]
        if cid >= 0:
            comm_deg[cid] = comm_deg.get(cid, 0.0) + degree[node_id]

    Q = 0.0
    for cid, d_c in comm_deg.items():
        m_c = intra.get(cid, 0.0)
        Q += (m_c / total_m) - (d_c / (2 * total_m)) ** 2
    return float(Q)