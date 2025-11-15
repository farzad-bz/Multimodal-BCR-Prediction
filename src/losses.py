import torch

# ======= Pairwise ranking loss (mini-batch friendly) =======
@torch.no_grad()
def _build_pair_mask(time, event):
    """
    time,event: tensors on device, shape [B], event in {0,1}
    Returns a boolean mask M over all pairs (i,j) where i had event and t_i < t_j.
    """
    # i must be an event; j any sample with later time
    event_i = (event == 1)[:, None]          # [B,1]
    later_j = (time[None, :] > time[:, None])  # [B,B]: j later than i (strict)
    return event_i & later_j


def pairwise_ranking_loss(risk, time, event):
    """
    risk,time,event: [B]; builds all valid comparable pairs within the batch
    Loss = -log sigma(r_i - r_j) averaged over valid pairs
    """
    M = _build_pair_mask(time, event)
    if not M.any():
        # zero with a gradient path
        return (risk * 0.0).sum()
    r_i = risk[:, None]
    r_j = risk[None, :]
    diffs = (r_i - r_j)[M]
    return -torch.log(torch.sigmoid(diffs) + 1e-12).mean()