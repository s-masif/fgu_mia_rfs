"""Federated training (uniform client-model averaging) and retrain-from-scratch
reference models.

Model selection uses VALIDATION accuracy only; the test set is never read during
training or checkpoint selection (R1). Client-level retraining drops the whole
forget client. Node-level retraining removes forget nodes from the supervised
objective only; the nodes remain in the transductive graph (agreed framing).
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models import PlainGCN, DATASET_CONFIGS, get_metrics, set_seed

# Unlearning protocol ratios (from the original config)
CLIENT_FORGET_RATIO = 0.20
NODE_FORGET_RATIO   = 0.10



def make_model(cfg: dict, data: dict, num_clients: int | None = None
               ) -> torch.nn.Module:
    """Build a fresh PlainGCN sized for the given (cfg, data).

    Plain 2-layer GCN — no per-client gates. Used for the retrain-from-scratch
    baseline (original, retrained, and shadow models) so the MIA-consistency
    experiment runs against a standard architecture with no custom
    personalization mechanism.

    The `num_clients` argument is accepted for API compatibility with existing
    callers (e.g. shadow training) but is ignored — PlainGCN has no gates
    indexed by client id.
    """
    return PlainGCN(
        in_features  = data["num_features"],
        hidden_dim   = cfg["params"]["hid_dim"],
        out_features = data["num_classes"],
        dropout      = cfg["params"]["dropout"],
    )


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


def federated_train(cfg: dict, clients: list, model: torch.nn.Module,
                    desc: str = "FedAvg") -> tuple[torch.nn.Module, float]:
    """Standard FedAvg loop with best-checkpoint selection by global val/test
    accuracy (via get_metrics). Returns (best_model, best_acc).
    """
    n_rounds    = cfg["params"]["num_rounds"]
    check_every = max(1, n_rounds // 10)

    best_acc, best_state = -1.0, None
    for rnd in range(n_rounds):
        client_models = []
        for cl in clients:
            cm = copy.deepcopy(model)
            train_client(cm, cl, cfg, use_momentum=True)
            client_models.append(cm)

        # FedAvg
        sd = model.state_dict()
        for key in sd:
            sd[key] = torch.stack(
                [cm.state_dict()[key].float() for cm in client_models]
            ).mean(0)
        model.load_state_dict(sd)

        if (rnd + 1) % check_every == 0 or rnd == n_rounds - 1:
            _, acc = get_metrics(model, clients, mask_type='val')
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_acc)


def define_forget_set_client(cfg: dict, n_clients: int, seed: int,
                              forget_ratio: float = CLIENT_FORGET_RATIO
                              ) -> list[int]:
    """Pick `forget_ratio` of clients at random."""
    rng = np.random.default_rng(seed)
    n_forget = max(1, int(round(forget_ratio * n_clients)))
    return sorted(rng.choice(n_clients, n_forget, replace=False).tolist())


def define_forget_set_node(clients: list, seed: int,
                            forget_ratio: float = NODE_FORGET_RATIO):
    """Pick one target client; sample `forget_ratio` of its training nodes.

    Returns (target_idx, forget_mask_local, retain_mask_local) where masks
    are bool arrays over the target client's LOCAL node indices.
    """
    rng = np.random.default_rng(seed)
    target_idx = int(rng.integers(0, len(clients)))
    tgt = clients[target_idx]
    train_idx = np.where(np.asarray(tgt["train_mask"]))[0]

    if len(train_idx) < 2:
        raise RuntimeError(
            f"Target client {target_idx} has only {len(train_idx)} train nodes; "
            "cannot sample forget set."
        )

    n_forget = max(1, int(round(forget_ratio * len(train_idx))))
    forget_local = rng.choice(train_idx, n_forget, replace=False)

    n_local = tgt["features"].shape[0]
    forget_mask_local = np.zeros(n_local, dtype=bool); forget_mask_local[forget_local] = True
    retain_mask_local = np.asarray(tgt["train_mask"]).copy() & ~forget_mask_local
    return target_idx, forget_mask_local, retain_mask_local


def retrain_gold_client(cfg, data, clients, forget_clients, seed):
    """Retrain FedAvg on retain clients only (forget clients excluded)."""
    set_seed(seed)
    retain = [c for i, c in enumerate(clients) if i not in forget_clients]
    # Keep cfg["num_clients"] the same so gate indexing stays aligned with
    # the original client IDs (forget clients' gates simply stay at init).
    model = make_model(cfg, data)
    return federated_train(cfg, retain, model, desc="Retrain[client]")


def retrain_gold_node(cfg, data, clients, target_idx, forget_mask_local, seed):
    """Retrain FedAvg from scratch with the forget nodes removed from the target
    client's supervised objective (their entries are dropped from train_mask).
    The graph is transductive: the forget nodes remain in the adjacency and
    still participate in message passing, so this is removal from supervision
    rather than complete removal of the data. All clients participate; only the
    target client's train_mask changes.
    """
    set_seed(seed)
    mod_clients = [copy.deepcopy(c) for c in clients]
    mod_clients[target_idx]["train_mask"] = (
        np.asarray(mod_clients[target_idx]["train_mask"]) & ~forget_mask_local
    )
    model = make_model(cfg, data)
    return federated_train(cfg, mod_clients, model, desc="Retrain[node]")