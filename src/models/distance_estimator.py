from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool

TWO_48_MINUS_1 = float(2**48 - 1)


class DistanceEstimator(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        use_goal: bool = True,
        node_emb_dim: int = 32,
        edge_emb_dim: int = 32,
    ):
        super().__init__()
        self.use_goal = use_goal

        # id_mlp: raw node ID scalar → node_emb_dim
        self.id_mlp = nn.Sequential(
            nn.Linear(1, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim),
        )

        # edge_mlp: raw edge_attr scalar → edge_emb_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )

        # GINEConv layers for state graph
        self.state_conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(node_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_emb_dim,
        )
        self.state_conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_emb_dim,
        )

        # GINEConv layers for goal graph (optional)
        if use_goal:
            self.goal_conv1 = GINEConv(
                nn.Sequential(
                    nn.Linear(node_emb_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=edge_emb_dim,
            )
            self.goal_conv2 = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=edge_emb_dim,
            )

        # final regressor: [graph_emb (± goal) + depth] → distance
        in_dim = hidden_dim * (2 if use_goal else 1) + 1
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, graph, conv1, conv2):
        # node names
        raw = graph.node_names.to(graph.edge_index.device).float()  # [N]
        norm = raw / TWO_48_MINUS_1

        norm = norm.clamp(0.0, 1.0)
        x = self.id_mlp(norm.view(-1, 1))  # [N,node_emb_dim]

        # edge features
        e_raw = graph.edge_attr.to(x.device).float()  # [E,1]
        e = self.edge_mlp(e_raw)  # [E,edge_emb_dim]

        # two GINEConv layers
        x = F.relu(conv1(x, graph.edge_index, e))
        x = F.relu(conv2(x, graph.edge_index, e))

        batch = graph.batch.clone()
        # graph‐level embedding
        return global_mean_pool(x, batch)

    def forward(self, batch_dict):
        s = self._encode(batch_dict["state_graph"], self.state_conv1, self.state_conv2)

        if self.use_goal and batch_dict.get("goal_graph") is not None:
            g = self._encode(batch_dict["goal_graph"], self.goal_conv1, self.goal_conv2)
            rep = torch.cat([s, g], dim=1)
        else:
            rep = s

        d = batch_dict["depth"].float().view(-1, 1)
        z = torch.cat([rep, d], dim=1)
        logits = self.regressor(z).squeeze(1)
        return logits

    def get_checkpoint(self) -> Dict:
        """
        Returns:
          - state_dict: model weights
          - config: init kwargs for reconstruction
        """
        cfg = {
            "hidden_dim": self.state_conv1.nn[0].out_features,
            "node_emb_dim": self.id_mlp[0].out_features,
            "edge_emb_dim": self.edge_mlp[0].out_features,
            "use_goal": self.use_goal,
        }
        return {"state_dict": self.state_dict(), "config": cfg}

    @classmethod
    def load_model(cls, ckpt: Dict):
        cfg = ckpt["config"]
        model = cls(
            hidden_dim=cfg["hidden_dim"],
            use_goal=cfg["use_goal"],
            node_emb_dim=cfg["node_emb_dim"],
            edge_emb_dim=cfg["edge_emb_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


class OnnxDistanceEstimatorWrapper(nn.Module):
    """
     Wraps `NewDistanceEstimator` so that ONNX sees only *plain tensors* –
     not PyG Data objects.  The wrapper re-implements _encode() for raw
     tensors, including a scatter-based global-mean-pool that is 100 % ONNX
    -export-able.
    """

    def __init__(self, core: "DistanceEstimator"):
        super().__init__()
        self.core = core  # the trained model √

    # --------------------------------------------------------------------- util
    @staticmethod
    def _global_mean(x, batch):
        """
                x     : [N, F]       (node embeddings)
                batch : [N]  (graph-id per node, 0 … B-1)
        1
                Pure-Torch implementation, ONNX-friendly, **no external scatter lib**.
        """
        # 1) accumulate sums per graph  ────────────────────────────────
        B = batch.max() + 1  # <— remains a Tensor!
        sums = torch.zeros(B, x.size(1), dtype=x.dtype, device=x.device)
        sums = sums.index_add(0, batch, x)  # [B, F]

        # 2) node counts per graph  ───────────────────────────────────
        ones = torch.ones_like(batch, dtype=x.dtype)
        cnts = torch.zeros(B, 1, dtype=x.dtype, device=x.device)
        cnts = cnts.index_add(0, batch, ones.unsqueeze(1))  # [B, 1]

        # 3) mean = sum / count  ( +ε avoids div-by-0 for empty graphs )
        return sums / cnts.clamp(min=1.0)

    # ----------------------------------------------------------------- encoder
    def _encode_raw(
        self, node_ids, edge_index, edge_attr, batch, conv1: nn.Module, conv2: nn.Module
    ):
        """
        A *tensor-only* version of NewDistanceEstimator._encode().
        Shapes are identical to the tensors fed from ONNX.
        """
        # 1) node scalar → [N, node_emb_dim]
        norm = torch.clamp((node_ids.float() + 2.0) / TWO_48_MINUS_1, 0.0, 1.0)
        x = self.core.id_mlp(norm.unsqueeze(1))

        # 2) edge scalar → [E, edge_emb_dim]
        e = self.core.edge_mlp(edge_attr.float())

        # 3) two GINEConv layers
        x = F.relu(conv1(x, edge_index, e))
        x = F.relu(conv2(x, edge_index, e))

        # 4) global mean pool → [B, hidden_dim]
        return self._global_mean(x, batch)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        s_node_ids: torch.Tensor,
        s_edge_index: torch.Tensor,
        s_edge_attr: torch.Tensor,
        s_batch: torch.Tensor,
        depth: torch.Tensor,
        g_node_ids: torch.Tensor,
        g_edge_index: torch.Tensor,
        g_edge_attr: torch.Tensor,
        g_batch: torch.Tensor,
    ) -> torch.Tensor:
        # ---------- state graph ----------
        s_emb = self._encode_raw(
            s_node_ids,
            s_edge_index,
            s_edge_attr,
            s_batch,
            self.core.state_conv1,
            self.core.state_conv2,
        )
        # ---------- goal graph (can be empty) ----------
        if self.core.use_goal and g_node_ids.numel() > 0:
            g_emb = self._encode_raw(
                g_node_ids,
                g_edge_index,
                g_edge_attr,
                g_batch,
                self.core.goal_conv1,
                self.core.goal_conv2,
            )
            rep = torch.cat([s_emb, g_emb], dim=1)  # [B, 2·H]
        else:
            rep = s_emb  # [B,   H]

        z = torch.cat([rep, depth.float().view(-1, 1)], dim=1)  # + depth
        return self.core.regressor(z).squeeze(1)  # → [B]
