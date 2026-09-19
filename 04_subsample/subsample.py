from __future__ import annotations

import numpy as np


def _build_adjacency_list(edge_index: np.ndarray, n: int) -> list[list[int]]:
    """edge_index: (2, E) or (E, 2). Returns undirected adjacency list."""
    ei = np.asarray(edge_index)
    if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.T
    adj: list[list[int]] = [[] for _ in range(n)]
    for s, d in zip(ei[0], ei[1]):
        s, d = int(s), int(d)
        if s == d:
            continue
        adj[s].append(d)
        adj[d].append(s)
    # dedupe
    return [list(set(nbrs)) for nbrs in adj]


def mhrw_sample_nodes(edge_index: np.ndarray, n: int, target_n: int,
                      seed: int = 0, max_steps_factor: int = 200) -> np.ndarray:
    """Sample `target_n` distinct nodes with a Metropolis-Hastings random walk.

    Returns a sorted array of sampled node indices. Handles getting stuck
    (low-degree pockets, disconnected components) by restarting from a fresh
    random seed node when the walk stalls.
    """
    rng = np.random.default_rng(seed)
    adj = _build_adjacency_list(edge_index, n)
    deg = np.array([len(a) for a in adj])

    # Nodes with no edges cannot be reached by a walk; sample them uniformly
    # at the end if needed so isolated nodes are not systematically excluded.
    connected = np.where(deg > 0)[0]
    if len(connected) == 0:
        # no edges at all: fall back to uniform node sampling
        return np.sort(rng.choice(n, size=min(target_n, n), replace=False))

    visited: set[int] = set()
    max_steps = max_steps_factor * target_n
    steps = 0
    stall = 0

    current = int(rng.choice(connected))
    visited.add(current)

    while len(visited) < target_n and steps < max_steps:
        steps += 1
        nbrs = adj[current]
        if not nbrs:
            current = int(rng.choice(connected)); visited.add(current); continue

        cand = int(rng.choice(nbrs))
        # MH acceptance: min(1, d_current / d_cand)
        if deg[cand] <= deg[current] or rng.random() < deg[current] / deg[cand]:
            current = cand
            before = len(visited)
            visited.add(current)
            stall = 0 if len(visited) > before else stall + 1
        else:
            stall += 1

        # If the walk stops adding new nodes for a while, jump to a new seed.
        if stall > 50:
            current = int(rng.choice(connected))
            visited.add(current)
            stall = 0

    sampled = np.array(sorted(visited))
    # If the walk could not reach enough nodes, top up uniformly at random
    # from the remaining nodes (reported; keeps target size honest).
    if len(sampled) < target_n:
        remaining = np.setdiff1d(np.arange(n), sampled)
        need = target_n - len(sampled)
        if need > 0 and len(remaining) > 0:
            extra = rng.choice(remaining, size=min(need, len(remaining)),
                               replace=False)
            sampled = np.sort(np.concatenate([sampled, extra]))
    return sampled


# ── Topology metrics on an induced subgraph ─────────────────────────────────

def _induced_edges(edge_index: np.ndarray, keep: np.ndarray):
    """Return edges (both endpoints in `keep`), remapped to 0..len(keep)-1."""
    ei = np.asarray(edge_index)
    if ei.ndim == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.T
    keep_set = set(int(x) for x in keep)
    remap = {int(old): i for i, old in enumerate(keep)}
    src, dst = [], []
    for s, d in zip(ei[0], ei[1]):
        s, d = int(s), int(d)
        if s in keep_set and d in keep_set and s != d:
            src.append(remap[s]); dst.append(remap[d])
    return np.array([src, dst]) if src else np.zeros((2, 0), dtype=int)


def topology_report(edge_index, labels, keep: np.ndarray) -> dict:
    """Measure homophily, average degree, and modularity of the induced
    subgraph on `keep`, plus node/edge counts. Used to verify that the
    subsample preserved topology."""
    n_sub = len(keep)
    ei_sub = _induced_edges(edge_index, keep)
    m = ei_sub.shape[1]  # directed count (each undirected edge appears... once here)

    y = np.asarray(labels)[keep]

    # Average degree (undirected): 2 * |E_undirected| / n.
    # _induced_edges lists each undirected edge once (from the original (2,E)
    # symmetric list it would be twice; we dedupe by building a set).
    und = set()
    for s, d in zip(ei_sub[0], ei_sub[1]):
        und.add((min(s, d), max(s, d)))
    n_edges = len(und)
    avg_deg = (2.0 * n_edges / n_sub) if n_sub > 0 else 0.0

    # Edge homophily: fraction of edges joining same-label nodes.
    if n_edges > 0:
        same = sum(1 for (s, d) in und if y[s] == y[d])
        homophily = same / n_edges
    else:
        homophily = float("nan")

    # Modularity of the label partition (communities = classes), a simple,
    # label-based modularity that is comparable across sizes.
    if n_edges > 0:
        deg = np.zeros(n_sub)
        for (s, d) in und:
            deg[s] += 1; deg[d] += 1
        two_m = 2.0 * n_edges
        Q = 0.0
        classes = np.unique(y)
        for c in classes:
            idx = np.where(y == c)[0]
            idx_set = set(idx.tolist())
            e_in = sum(1 for (s, d) in und if s in idx_set and d in idx_set)
            d_c = deg[idx].sum()
            Q += (e_in / n_edges) - (d_c / two_m) ** 2
    else:
        Q = float("nan")

    return {
        "sub_nodes": int(n_sub),
        "sub_edges": int(n_edges),
        "sub_avg_degree": float(avg_deg),
        "sub_homophily": float(homophily),
        "sub_modularity_labels": float(Q),
        "sub_n_classes": int(len(np.unique(y))),
    }
