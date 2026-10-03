"""Generalization gap and utility metrics, with the corrected definition (R5).

The gap is acc_train(MR) - acc_test(MR), where training accuracy is computed on
the RETAINED supervised mask R = S \\ F (forget nodes excluded from the
training-accuracy denominator) and test accuracy on the untouched test mask. The
gap is treated as a possible confounder, not a predetermined cause.
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score


@torch.no_grad()
def _masked_accuracy(model, clients, mask_key, exclude_local=None):
    """Mean over clients of accuracy on the given mask.
    `exclude_local` optionally maps client index -> boolean local mask of nodes
    to drop from the numerator/denominator (used to exclude forget nodes from the
    training-accuracy computation in the node scenario).
    """
    model.eval()
    all_true, all_pred = [], []
    for i, c in enumerate(clients):
        mask = np.asarray(c[mask_key]).copy()
        if exclude_local is not None and i in exclude_local:
            mask = mask & ~exclude_local[i]
        if mask.sum() == 0:
            continue
        out = model(c["features"], c["adj"], i)
        pred = out[mask].argmax(1).cpu().numpy()
        true = c["labels"][mask].cpu().numpy()
        all_true.extend(true)
        all_pred.extend(pred)
    return accuracy_score(all_true, all_pred) if all_true else float("nan")


def generalization_gap(model_ret, clients, scenario, forget_info):
    """Corrected train/test gap on the retrained model.

    scenario == 'client': forget clients are already absent from `clients`
        passed here (retain clients only), so training accuracy is over the
        retained supervised nodes by construction.
    scenario == 'node'  : forget nodes must be excluded from the training-accuracy
        mask of the target client.
    Returns (acc_train, acc_test, gap).
    """
    exclude = None
    if scenario == "node" and forget_info is not None:
        target_idx, forget_mask_local = forget_info[0], forget_info[1]
        exclude = {int(target_idx): np.asarray(forget_mask_local)}

    acc_train = _masked_accuracy(model_ret, clients, "train_mask", exclude_local=exclude)
    acc_test  = _masked_accuracy(model_ret, clients, "test_mask")
    gap = acc_train - acc_test
    return float(acc_train), float(acc_test), float(gap)