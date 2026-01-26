import torch
import torch.nn.functional as F
def _ensure_1d(x): # torch.Tensor -> torch.Tensor
    return x.squeeze()
def _to_labels01(targets): # torch.Tensor -> torch.Tensor 
    t = targets.squeeze()
    if t.dtype.is_floating_point:
        # keep -1 as-is for masking upstream; convert others
        t01 = t.clone()
        mask_neg1 = t01 == -1
        t01[~mask_neg1] = (t01[~mask_neg1] > 0.5).float()
        return t01.long()
    else:
        # integer types
        if torch.any((t != 0) & (t != 1) & (t != -1)):
            # maybe {-1,1}
            return ((t == 1).long())
        return t.long()
def _apply_class_weights(per_sample_loss, targets01, class_weight):
    if class_weight is None:
        return per_sample_loss
    w_neg, w_pos = float(class_weight[0]), float(class_weight[1])
    w = torch.where(targets01 == 1, torch.tensor(w_pos, device=per_sample_loss.device), torch.tensor(w_neg, device=per_sample_loss.device))
    return per_sample_loss * w
def focal_loss(preds, targets, gamma=2.0, alpha=0.25, logits=False, 
               eps=1e-6, return_per_sample=False, class_weight=None, 
               mask=None, reduction="mean"): 
    if preds.dim() == 0:
        raise ValueError("preds must be 1-D tensor of shape (B,) or (B,1)")
    if targets.dim() == 0:
        raise ValueError("targets must be 1-D tensor of shape (B,)")
    preds = _ensure_1d(preds).float()
    targets01 = _to_labels01(targets)
    include = torch.ones_like(targets01, dtype=torch.bool, device=preds.device) if mask is None else mask.bool()
    if include.sum() == 0:
        per = torch.zeros(preds.shape[0], device=preds.device)
        return per if return_per_sample else per.mean()
    logits = preds.clamp(eps,1-eps) if logits else torch.log(preds.clamp(eps,1-eps) / (1 - preds.clamp(eps,1-eps)))
    probs = torch.sigmoid(logits)
    p_t = torch.where(targets01 == 1, probs, 1.0 - probs)
    bce_per = F.binary_cross_entropy_with_logits(logits, targets01.float(), reduction="none")
    focal_factor = (1.0 - p_t) ** gamma
    loss_per = focal_factor * bce_per
    if alpha is not None:
        alpha_t = torch.where(targets01 == 1, torch.tensor(alpha, device=loss_per.device), torch.tensor(1.0 - alpha, device=loss_per.device))
        loss_per = loss_per * alpha_t
    loss_per = _apply_class_weights(loss_per, targets01, class_weight)
    # zero out excluded samples
    loss_out = torch.zeros_like(loss_per)
    loss_out[include] = loss_per[include]
    if return_per_sample:
        return loss_out
    if reduction == "mean":
        denom = include.sum().float().clamp(min=1.0)
        return loss_out.sum() / denom
    if reduction == "sum":
        return loss_out.sum()
    return loss_out