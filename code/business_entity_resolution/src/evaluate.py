"""Macro F0.5 exactly as described in the challenge: per Source 1 entity, averaged over all of them."""


def f05_entity(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, true: dict, ids) -> float:
    ids = list(ids)
    return sum(f05_entity(pred.get(i, set()), true.get(i, set())) for i in ids) / max(1, len(ids))
