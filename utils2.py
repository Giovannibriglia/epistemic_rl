import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _to_device(d, device):
    """Move a Data / HeteroData object to CUDA/CPU recursively."""
    return d.to(device, non_blocking=True)


def _ensure_batch_field(d):
    """
    PyG Data/HeteroData created by hand lack the .batch vectors
    expected by global_mean_pool.  DataLoader adds them automatically,
    but for single‑graph inference we add them here.
    """
    if isinstance(d, Data):
        if not hasattr(d, "batch"):
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=d.x.device)
    else:  # HeteroData
        for node_type in d.node_types:
            store = d[node_type]
            if not hasattr(store, "batch"):
                store.batch = torch.zeros(
                    store.num_nodes, dtype=torch.long, device=store.x.device
                )
    return d


# --------------------------------------------------------------------
# single‑sample prediction ---------------------------------------
# --------------------------------------------------------------------
def predict_single(sample_graph, model, device="cpu"):
    """
    sample_graph : G_s (+ G_g if USE_GOAL)   →  PyG Data/HeteroData
    model        : trained StateOnlyGNN or HeteroGNN
    returns      : scalar prediction (Python float)
    """
    model.eval()
    with torch.no_grad():
        g = _ensure_batch_field(_to_device(sample_graph, device))
        out = model(g)  # model knows whether g is Data or HeteroData
        return out.item()


# --------------------------------------------------------------------
# batched prediction ---------------------------------------------
# --------------------------------------------------------------------


def predict_batch(graph_list, model, batch_size=128, device="cpu"):
    """
    graph_list : list[Data] or list[HeteroData]
    returns    : torch.Tensor of predictions, shape [N]
    """
    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            preds.append(model(batch).cpu())
    return torch.cat(preds)
