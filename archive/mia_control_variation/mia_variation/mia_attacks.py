"""
mia_attacks.py
==============
Two membership-inference attacks, faithful to Shokri et al. (S&P 2017):

  1. ShokriShadowMIA — shadow models + per-class attack classifiers, with
                       softmax-probability features. PyTorch port of
                       csong27/membership-inference (attack.py + classifier.py).

  2. LRConfidenceMIA — lightweight logistic regression on confidence
                       features (max_conf, entropy, margin, true_conf).
                       Fast comparison attack.

Both produce a comprehensive metrics dict:
  • Overall MIA           : accuracy, precision, recall, F1, AUC
  • Per membership class  : member / non-member precision, recall, F1
  • Per data class        : member_acc, precision, recall, F1, n
  • Subset breakdowns     : predicted_member_rate on forget/retain/test

Reference: https://github.com/csong27/membership-inference

NOTE on two corrections to the naive port (see ShokriShadowMIA docstring):
  Fix 1 — class-weighted loss in the attack classifier, so a 2:1 member:
          non-member imbalance in the shadow data cannot be minimised by
          always predicting "member".
  Fix 2 — per-class decision threshold calibrated on SHADOW non-member scores
          at a target false-positive rate, replacing the hard argmax-at-0.5
          rule. `member_score` (hence AUC) is unchanged by Fix 2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, roc_auc_score,
    balanced_accuracy_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler


# ════════════════════════════════════════════════════════════════════════════
#  PyTorch port of csong27/membership-inference/classifier.py
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
#  Shokri Shadow MIA  (per-class attack models)
# ════════════════════════════════════════════════════════════════════════════

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
                 fpr: float = 0.1):
        self.cfg = dict(attack_type=attack_type, n_hidden=n_hidden,
                        epochs=epochs, batch_size=batch_size,
                        lr=lr, l2_ratio=l2_ratio)
        self.fpr = float(fpr)
        self.class_models: dict[int, nn.Module | None] = {}
        self.class_thresholds: dict[int, float] = {}      # Fix 2
        self.fallback_model: nn.Module | None = None
        self.fallback_threshold: float = 0.5              # Fix 2
        self.n_in: int | None = None

    # ── Fit on shadow data ────────────────────────────────────────────────
    def fit(self,
            shadow_probs:   np.ndarray,    # (N, C) softmax probs
            shadow_member:  np.ndarray,    # (N,)   {0,1}
            shadow_classes: np.ndarray,    # (N,)   true class labels
            n_classes:      int) -> "ShokriShadowMIA":
        self.n_in = shadow_probs.shape[1]

        # Per-class attack models (Shokri's design)
        for c in range(n_classes):
            sel = shadow_classes == c
            n_c = int(sel.sum())
            uniq = np.unique(shadow_member[sel]) if n_c > 0 else np.array([])
            if n_c < 10 or len(uniq) < 2:
                self.class_models[c] = None
                continue
            model = _train_attack_clf(
                shadow_probs[sel], shadow_member[sel], **self.cfg
            )
            self.class_models[c] = model
            # Fix 2: calibrate this class's threshold on its OWN shadow
            # non-members (in-sample scores from the model just trained).
            neg = shadow_probs[sel][shadow_member[sel] == 0]
            neg_scores = _score_attack_clf(model, neg) if len(neg) else np.array([])
            self.class_thresholds[c] = _threshold_at_fpr(neg_scores, self.fpr)

        # Fallback model — pooled across all classes (for sparse classes)
        if any(m is None for m in self.class_models.values()):
            self.fallback_model = _train_attack_clf(
                shadow_probs, shadow_member, **self.cfg
            )
            neg = shadow_probs[shadow_member == 0]
            neg_scores = (_score_attack_clf(self.fallback_model, neg)
                          if len(neg) else np.array([]))
            self.fallback_threshold = _threshold_at_fpr(neg_scores, self.fpr)
        return self

    # ── Apply to target data ──────────────────────────────────────────────
    def predict(self,
                query_probs:   np.ndarray,
                query_classes: np.ndarray):
        """For each query node, apply its class's attack model.

        Membership is decided by comparing the member-score against the
        class's calibrated threshold (Fix 2), not by argmax-at-0.5.

        Returns
        -------
        pred           : (N,) int   predicted membership ∈ {0, 1}
        member_score   : (N,) float member probability
        used_fallback  : (N,) bool  whether the fallback model was used
        """
        N = len(query_probs)
        pred          = np.zeros(N, dtype=int)
        member_score  = np.full(N, 0.5, dtype=float)
        used_fallback = np.zeros(N, dtype=bool)

        for c, model in self.class_models.items():
            sel = query_classes == c
            if not sel.any():
                continue
            if model is None:
                if self.fallback_model is None:
                    continue
                s   = _score_attack_clf(self.fallback_model, query_probs[sel])
                thr = self.fallback_threshold
                used_fallback[sel] = True
            else:
                s   = _score_attack_clf(model, query_probs[sel])
                thr = self.class_thresholds.get(c, 0.5)
            member_score[sel] = s
            pred[sel] = (s > thr).astype(int)

        # Cover query classes never seen in shadow data (shouldn't happen often)
        unseen = ~np.isin(query_classes, list(self.class_models.keys()))
        if unseen.any() and self.fallback_model is not None:
            s = _score_attack_clf(self.fallback_model, query_probs[unseen])
            member_score[unseen] = s
            pred[unseen] = (s > self.fallback_threshold).astype(int)
            used_fallback[unseen] = True

        return pred, member_score, used_fallback


# ════════════════════════════════════════════════════════════════════════════
#  Lightweight LR confidence-feature MIA (comparison)
# ════════════════════════════════════════════════════════════════════════════

class LRConfidenceMIA:
    """Logistic regression on confidence features.

    Features per query: [max_conf, entropy, margin, true_class_conf]. Trained
    on (target_probs[member], target_probs[non-member]) where member labels
    are derived from the target model's known retain/test split.

    Fix 1 is applied here too via `class_weight='balanced'`, for the same
    reason as the Shokri attack: the member/non-member training labels are
    imbalanced and otherwise collapse the classifier to the majority class.
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        self.lr     = LogisticRegression(max_iter=max_iter, C=C,
                                          class_weight="balanced")
        self.scaler = StandardScaler()

    @staticmethod
    def _features(probs: np.ndarray, true_classes: np.ndarray) -> np.ndarray:
        eps = 1e-12
        max_conf  = probs.max(axis=1)
        entropy   = -(probs * np.log(probs + eps)).sum(axis=1)
        sorted_p  = np.sort(probs, axis=1)
        margin    = sorted_p[:, -1] - sorted_p[:, -2]
        true_conf = probs[np.arange(len(probs)), true_classes]
        return np.stack([max_conf, entropy, margin, true_conf], axis=1)

    def fit(self, probs: np.ndarray, member: np.ndarray,
            true_classes: np.ndarray) -> "LRConfidenceMIA":
        X = self.scaler.fit_transform(self._features(probs, true_classes))
        self.lr.fit(X, member)
        return self

    def predict(self, probs: np.ndarray, true_classes: np.ndarray):
        X = self.scaler.transform(self._features(probs, true_classes))
        pred  = self.lr.predict(X)
        score = self.lr.predict_proba(X)[:, 1]
        return pred, score


# ════════════════════════════════════════════════════════════════════════════
#  Detailed MIA metrics
# ════════════════════════════════════════════════════════════════════════════

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
            m["mia_tpr_at_fpr_0.1"]   = _tpr_at(0.1)
        except ValueError:
            m["mia_tpr_at_fpr_0.001"] = float("nan")
            m["mia_tpr_at_fpr_0.01"]  = float("nan")
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