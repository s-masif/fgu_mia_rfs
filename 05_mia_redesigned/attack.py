import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             precision_recall_fscore_support,
                             balanced_accuracy_score)

ALPHA = 0.10   # fixed operating point (false-positive rate) for the decision threshold


def assert_pools_disjoint(fit_ids, calib_ids, eval_ids):
    """Fail fast if any evaluation node identity leaks into fit/calibration.
    All arguments are iterables of global node IDs."""
    fit_s, calib_s, eval_s = set(map(int, fit_ids)), set(map(int, calib_ids)), set(map(int, eval_ids))
    leak_fit   = eval_s & fit_s
    leak_calib = eval_s & calib_s
    assert not leak_fit,   f"{len(leak_fit)} evaluation node(s) present in the FIT pool"
    assert not leak_calib, f"{len(leak_calib)} evaluation node(s) present in the CALIBRATION pool"


def _threshold_at_fpr(neg_scores: np.ndarray, fpr: float) -> float:
    """Pick the member-score threshold giving ≈`fpr` false positives on a set
    of KNOWN non-members (neg_scores). A query is predicted "member" if its
    score is strictly greater than this threshold.

    Implemented as the (1 - fpr) empirical quantile of the non-member scores:
    only an `fpr` fraction of non-members exceed it. Falls back to 0.5 if there
    are no non-member scores to calibrate on.
    """
    if neg_scores.size == 0:
        return 0.5
    fpr = float(np.clip(fpr, 0.0, 1.0))
    return float(np.quantile(neg_scores, 1.0 - fpr))


class _ShokriNN(nn.Module):
    """Shokri 'nn' attack model — 1 hidden layer, tanh, softmax output.

    Direct port of classifier.py:get_nn_model.
    """
    def __init__(self, n_in: int, n_hidden: int = 50, n_out: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss applies softmax internally → return logits
        return self.fc2(torch.tanh(self.fc1(x)))


class _ShokriSoftmax(nn.Module):
    """Shokri 'softmax' attack model — logistic regression with softmax output.

    Direct port of classifier.py:get_softmax_model. This is the DEFAULT
    attack_model in the reference attack.py.
    """
    def __init__(self, n_in: int, n_out: int = 2):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _build_attack_clf(attack_type: str, n_in: int, n_hidden: int) -> nn.Module:
    if attack_type == "nn":
        return _ShokriNN(n_in, n_hidden, n_out=2)
    if attack_type == "softmax":
        return _ShokriSoftmax(n_in, n_out=2)
    raise ValueError(f"Unknown attack_type {attack_type!r}")


def _train_attack_clf(
    X: np.ndarray, y: np.ndarray,
    attack_type: str = "nn", n_hidden: int = 50,
    epochs: int = 100, batch_size: int = 100,
    lr: float = 0.01, l2_ratio: float = 1e-7,
) -> nn.Module:
    """Port of classifier.py:train.

    Adam + categorical cross-entropy + L2 weight decay (Shokri's setup).

    Fix 1: the cross-entropy loss is class-weighted by inverse membership-label
    frequency. With ~2:1 member:non-member shadow data and weakly-separable
    features, an unweighted loss is minimised by predicting the majority
    ("member") for everything; the weights remove that shortcut so the model is
    forced to use whatever membership signal exists.
    """
    n_in = X.shape[1]
    model = _build_attack_clf(attack_type, n_in, n_hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_ratio)

    # Fix 1: inverse-frequency class weights over membership labels {0, 1}.
    counts = np.bincount(y, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    n   = len(X_t)
    bs  = min(batch_size, n)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _score_attack_clf(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return member probabilities (column 1 of the softmax) only."""
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return F.softmax(logits, dim=1).cpu().numpy()[:, 1]



class ShokriShadowMIA:
    """Shokri et al. (S&P 2017) shadow-model MIA with per-class attack models.

    Faithfully follows csong27/membership-inference/attack.py:train_attack_model:
    one attack classifier is trained PER true class label of the underlying
    classification task, using softmax-probability vectors as features and
    binary {0=non-member, 1=member} labels as targets.

    A small fallback model trained on the full pooled shadow data is used for
    classes with insufficient data in the shadow set (e.g. < 10 samples or only
    one membership label present).

    Two corrections to the naive port:
      Fix 1 (in _train_attack_clf) — class-weighted loss, so a 2:1 member
            imbalance plus weak features cannot collapse the attack to an
            all-"member" predictor.
      Fix 2 (here) — the member/non-member decision uses a per-class threshold
            calibrated on SHADOW non-member scores at a target false-positive
            rate `fpr`, instead of the hard argmax-at-0.5 rule. The continuous
            `member_score` is returned unchanged, so AUC is unaffected by Fix 2
            and isolates the effect of Fix 1.
    """

    def __init__(self,
                 attack_type: str = "nn", n_hidden: int = 50,
                 epochs: int = 100, batch_size: int = 100,
                 lr: float = 0.01, l2_ratio: float = 1e-7,
                 fpr: float = 0.1, alpha: float | None = None):
        self.cfg = dict(attack_type=attack_type, n_hidden=n_hidden,
                        epochs=epochs, batch_size=batch_size,
                        lr=lr, l2_ratio=l2_ratio)
        # `alpha` (the fixed operating point) takes precedence if given.
        self.fpr = float(alpha) if alpha is not None else float(fpr)
        self.alpha = self.fpr
        self.class_models: dict[int, nn.Module | None] = {}
        self.class_thresholds: dict[int, float] = {}      # Fix 2
        self.fallback_model: nn.Module | None = None
        self.fallback_threshold: float = 0.5              # Fix 2
        self.fallback_used: dict[int, bool] = {}          # per-class fallback flag
        self.n_in: int | None = None

    # ── Fit on shadow data ────────────────────────────────────────────────
    def fit(self,
            shadow_probs:   np.ndarray,      # (N,C) fit-pool softmax
            shadow_member:  np.ndarray,      # (N,)  {0,1}
            shadow_classes: np.ndarray,      # (N,)  true class labels
            n_classes:      int | None = None,
            calib_probs:    np.ndarray | None = None,   # separate calibration pool
            calib_member:   np.ndarray | None = None,
            calib_cls:      np.ndarray | None = None,
            min_calib:      int = 10) -> "ShokriShadowMIA":
        """Fit the attack.

        Policy (Point 5):
          * ALWAYS train a pooled fallback attack on all classes together, and
            calibrate its threshold at alpha on the pooled calibration
            non-members.
          * For each class: if it has enough fit data AND enough calibration
            non-members (>= min_calib), fit a class-specific attack and
            calibrate its own threshold at alpha on that class's calibration
            non-members. Otherwise the class is routed to the POOLED attack
            together with the POOLED threshold, and this is recorded in
            fallback_used[c] = True.
        The class-specific attack is never given a pooled threshold, and a
        sparse class is never left on an uncalibrated class-specific attack.
        """
        self.n_in = shadow_probs.shape[1]
        if n_classes is None:
            mx = int(shadow_classes.max()) if len(shadow_classes) else 0
            if calib_cls is not None and len(calib_cls):
                mx = max(mx, int(calib_cls.max()))
            n_classes = mx + 1

        use_oos = (calib_probs is not None and calib_member is not None
                   and calib_cls is not None and len(calib_cls) > 0)
        if not use_oos:
            raise ValueError("A separate calibration pool is required (Point 5).")

        # ── Pooled fallback attack: always trained, always calibrated ─────
        self.fallback_model = _train_attack_clf(shadow_probs, shadow_member, **self.cfg)
        fneg = calib_probs[calib_member == 0]
        fneg_scores = _score_attack_clf(self.fallback_model, fneg) if len(fneg) else np.array([])
        self.fallback_threshold = _threshold_at_fpr(fneg_scores, self.alpha)
        self.fallback_calib_n = int(len(fneg))

        # ── Per-class attacks, only where fit + calibration support exists ─
        self.class_models = {}
        self.class_thresholds = {}
        self.fallback_used = {}
        self.calib_support = {}
        for c in range(n_classes):
            sel = shadow_classes == c
            n_c = int(sel.sum())
            uniq = np.unique(shadow_member[sel]) if n_c > 0 else np.array([])
            cneg = calib_probs[(calib_cls == c) & (calib_member == 0)]
            n_cal = int(len(cneg))
            self.calib_support[c] = n_cal

            enough_fit   = (n_c >= 10 and len(uniq) >= 2)
            enough_calib = (n_cal >= min_calib)

            if enough_fit and enough_calib:
                model = _train_attack_clf(shadow_probs[sel], shadow_member[sel], **self.cfg)
                neg_scores = _score_attack_clf(model, cneg)
                self.class_models[c] = model
                self.class_thresholds[c] = _threshold_at_fpr(neg_scores, self.alpha)
                self.fallback_used[c] = False
            else:
                # Route this class to the pooled attack + pooled threshold.
                self.class_models[c] = None
                self.class_thresholds[c] = self.fallback_threshold
                self.fallback_used[c] = True
        return self

    # ── Apply to target data ──────────────────────────────────────────────
    def predict(self,
                query_probs:   np.ndarray,
                query_classes: np.ndarray):
        """Score query nodes with the frozen attack.

        A class that was routed to the fallback at fit time is scored with the
        POOLED attack and the POOLED threshold; otherwise with its own attack
        and its own threshold. Nothing is recalibrated here (R4).

        Returns
        -------
        pred           : (N,) int   predicted membership in {0, 1}
        member_score   : (N,) float member probability
        thr_used       : (N,) float threshold applied to each node
        fb_used        : (N,) bool  True where the pooled fallback attack was used
        """
        N = len(query_probs)
        pred         = np.zeros(N, dtype=int)
        member_score = np.full(N, 0.5, dtype=float)
        thr_used     = np.full(N, self.fallback_threshold, dtype=float)
        fb_used      = np.zeros(N, dtype=bool)

        seen = np.zeros(N, dtype=bool)
        for c, model in self.class_models.items():
            sel = query_classes == c
            if not sel.any():
                continue
            seen |= sel
            if model is None or self.fallback_used.get(c, False):
                s_   = _score_attack_clf(self.fallback_model, query_probs[sel])
                thr  = self.fallback_threshold
                fb_used[sel] = True
            else:
                s_   = _score_attack_clf(model, query_probs[sel])
                thr  = self.class_thresholds[c]
            member_score[sel] = s_
            thr_used[sel]     = thr
            pred[sel]         = (s_ > thr).astype(int)

        # Query classes never seen at fit time -> pooled fallback, recorded.
        unseen = ~seen
        if unseen.any():
            s_ = _score_attack_clf(self.fallback_model, query_probs[unseen])
            member_score[unseen] = s_
            thr_used[unseen]     = self.fallback_threshold
            pred[unseen]         = (s_ > self.fallback_threshold).astype(int)
            fb_used[unseen]      = True

        return pred, member_score, thr_used, fb_used


def compute_detailed_mia_metrics(
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    y_score:       np.ndarray,
    true_classes:  np.ndarray,
    subset_masks:  dict | None = None,
) -> dict:
    """Comprehensive MIA metrics — overall, per membership class, per data
    class, and per subset (forget/retain/test).

    Parameters
    ----------
    y_true        : (N,) ground-truth membership labels (0/1)
    y_pred        : (N,) predicted membership labels  (0/1)
    y_score       : (N,) predicted member probability (for AUC)
    true_classes  : (N,) true data-class labels (used for per-class breakdown)
    subset_masks  : optional dict {'forget': bool_mask, 'retain': ..., 'test': ...}

    Returns
    -------
    dict containing all requested metrics. Nested 'per_data_class' maps class
    label → {acc, member_acc, member_p/r/f1, nonmember_p/r/f1, n_member, ...}.
    """
    m: dict = {}

    # ── Overall binary task ───────────────────────────────────────────────
    m["mia_acc"]       = float(accuracy_score(y_true, y_pred))
    m["mia_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    m["mia_recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    m["mia_f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_true)) == 2:
        try:
            m["mia_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            m["mia_auc"] = float("nan")
    else:
        m["mia_auc"] = float("nan")

    # ── Robust-to-imbalance and Carlini-2022 metrics ──────────────────────
    # Balanced accuracy = (TPR + TNR) / 2. Robust to class imbalance in the
    # query pool (Acc alone is dominated by the majority class when pool
    # sizes differ).
    try:
        m["mia_balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        m["mia_balanced_acc"] = float("nan")

    # TPR @ low FPR — following Carlini et al., "Membership Inference Attacks
    # From First Principles" (S&P 2022): average AUC hides the attack's
    # behaviour in the low-FPR regime, which is the regime that actually
    # matters for a real attacker. Report TPR at three thresholds.
    if len(np.unique(y_true)) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            def _tpr_at(target_fpr: float) -> float:
                below = fpr <= target_fpr
                return float(tpr[below].max()) if below.any() else 0.0
            m["mia_tpr_at_fpr_0.001"] = _tpr_at(0.001)
            m["mia_tpr_at_fpr_0.01"]  = _tpr_at(0.01)
            m["mia_tpr_at_fpr_0.05"]  = _tpr_at(0.05)
            m["mia_tpr_at_fpr_0.1"]   = _tpr_at(0.1)
        except ValueError:
            m["mia_tpr_at_fpr_0.001"] = float("nan")
            m["mia_tpr_at_fpr_0.01"]  = float("nan")
            m["mia_tpr_at_fpr_0.05"]  = float("nan")
            m["mia_tpr_at_fpr_0.1"]   = float("nan")
    else:
        m["mia_tpr_at_fpr_0.001"] = float("nan")
        m["mia_tpr_at_fpr_0.01"]  = float("nan")
        m["mia_tpr_at_fpr_0.1"]   = float("nan")

    # ── Per membership class (member, non-member) ─────────────────────────
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    m["nonmember_precision"] = float(p[0])
    m["nonmember_recall"]    = float(r[0])
    m["nonmember_f1"]        = float(f[0])
    m["nonmember_support"]   = int(sup[0])
    m["member_precision"]    = float(p[1])
    m["member_recall"]       = float(r[1])
    m["member_f1"]           = float(f[1])
    m["member_support"]      = int(sup[1])

    # ── Per data class ────────────────────────────────────────────────────
    per_class: dict = {}
    for c in np.unique(true_classes):
        sel = true_classes == c
        if sel.sum() < 2:
            continue
        yt, yp = y_true[sel], y_pred[sel]
        entry = {
            "n":            int(sel.sum()),
            "n_member":     int((yt == 1).sum()),
            "n_nonmember":  int((yt == 0).sum()),
            "acc":          float(accuracy_score(yt, yp)),
        }
        if len(np.unique(yt)) == 2:
            pc, rc, fc, _ = precision_recall_fscore_support(
                yt, yp, labels=[0, 1], zero_division=0)
            entry.update(
                nonmember_p  = float(pc[0]),
                nonmember_r  = float(rc[0]),
                nonmember_f1 = float(fc[0]),
                member_p     = float(pc[1]),
                member_r     = float(rc[1]),
                member_f1    = float(fc[1]),
                member_acc   = float(yp[yt == 1].mean()) if (yt == 1).any()
                               else float("nan"),
            )
        per_class[int(c)] = entry
    m["per_data_class"] = per_class

    # ── Subset breakdowns (the headline) ──────────────────────────────────
    if subset_masks:
        for name, mask in subset_masks.items():
            n_sub = int(mask.sum())
            if n_sub == 0:
                m[f"{name}_n"] = 0
                continue
            m[f"{name}_n"]                     = n_sub
            m[f"{name}_predicted_member_rate"] = float(y_pred[mask].mean())
            m[f"{name}_mean_score"]            = float(y_score[mask].mean())
    return m
