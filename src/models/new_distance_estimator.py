from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool


class NewDistanceEstimator(nn.Module):
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
        """# 1) node features: normalize raw integer IDs
        raw = graph.node_names.to(graph.edge_index.device).float()  # [N]
        mean, std = raw.mean(), raw.std(unbiased=False).clamp(min=1e-6)
        norm = (raw - mean) / std"""
        raw = graph.node_names.to(graph.edge_index.device).float()  # [N]
        # TODO: remove
        raw += 2
        norm = raw / (2**48 - 1)

        norm = norm.clamp(0.0, 1.0)
        x = self.id_mlp(norm.view(-1, 1))  # [N,node_emb_dim]
        """x = self.id_mlp(
            graph.node_names.to(graph.edge_index.device).float().view(-1, 1)
        )"""
        # 2) edge features
        e_raw = graph.edge_attr.to(x.device).float()  # [E,1]
        e = self.edge_mlp(e_raw)  # [E,edge_emb_dim]

        # 3) two GINEConv layers
        x = F.relu(conv1(x, graph.edge_index, e))
        x = F.relu(conv2(x, graph.edge_index, e))

        batch = graph.batch.clone()
        # 4) graph‐level embedding
        return global_mean_pool(x, batch)

    def forward(self, batch_dict):
        s = self._encode(batch_dict["state_graph"], self.state_conv1, self.state_conv2)

        if self.use_goal and batch_dict.get("goal_graph") is not None:
            g = self._encode(batch_dict["goal_graph"], self.goal_conv1, self.goal_conv2)
            rep = torch.cat([s, g], dim=1)
        else:
            rep = s

        """# incorporate normalized depth
        d = batch_dict["depth"].float().view(-1, 1)
        d = (d - d.mean()) / (d.std(unbiased=False) + 1e-6)"""
        # incorporate depth (assumed pre-normalized)
        d = batch_dict["depth"].float().view(-1, 1)
        z = torch.cat([rep, d], dim=1)
        """print("  [DEBUG] s_embed:", s.shape, s)  # after state‐graph pooling
        if self.use_goal:
            print("  [DEBUG] g_embed:", g.shape, g)
        print("  [DEBUG] depth  norm:", d.shape, d)
        print("  [DEBUG] concat z:", z.shape, z)"""
        return self.regressor(z).squeeze(1)  # raw distance

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
