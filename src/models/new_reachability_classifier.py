from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool


class NewReachabilityClassifier(nn.Module):
    """
    A reachability classifier based on GINEConv that predicts
    the probability of a target feature (binary classification).
    """

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

        # GINEConv layers for goal graph (if used)
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

        # final classifier: [graph_emb (+ goal_emb) + depth] → distance
        in_dim = hidden_dim * (2 if use_goal else 1) + 1
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, graph, conv1, conv2):
        # 1) node features
        raw = graph.node_names.to(graph.edge_index.device).float()  # [N]
        m = raw.mean(dim=0, keepdim=True)
        s = raw.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
        norm = (raw - m) / s
        x = self.id_mlp(norm.view(-1, 1))  # [N, node_emb_dim]

        # 2) edge features
        e_raw = graph.edge_attr.to(x.device).float()  # [E,1]
        e = self.edge_mlp(e_raw)  # [E, edge_emb_dim]

        # 3) two GINE layers
        x = F.relu(conv1(x, graph.edge_index, e))
        x = F.relu(conv2(x, graph.edge_index, e))

        batch = graph.batch.clone()
        # 4) graph‐level embedding
        return global_mean_pool(x, batch)

    def forward(self, batch_dict):
        # state graph embedding
        s = self._encode(
            batch_dict["state_graph"],
            self.state_conv1,
            self.state_conv2,
        )

        # goal graph embedding if used
        if self.use_goal and batch_dict.get("goal_graph") is not None:
            g = self._encode(
                batch_dict["goal_graph"],
                self.goal_conv1,
                self.goal_conv2,
            )
            rep = torch.cat([s, g], dim=1)
        else:
            rep = s

        # depth feature
        depth = batch_dict["depth"].float().view(-1, 1)
        depth = (depth - depth.mean()) / (depth.std(unbiased=False) + 1e-6)

        # classify
        z = torch.cat([rep, depth], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits

    def get_checkpoint(self) -> Dict:
        """
        Returns a dict containing:
          - state_dict: the model’s weights
          - config: arguments needed to reinstantiate this exact architecture
        """
        # First linear of state_conv1’s MLP has out_features == hidden_dim
        hidden_dim = self.state_conv1.nn[0].out_features
        # First linear of id_mlp maps 1 → node_emb_dim
        node_emb_dim = self.id_mlp[0].out_features
        # First linear of edge_mlp maps 1 → edge_emb_dim
        edge_emb_dim = self.edge_mlp[0].out_features
        use_goal = self.use_goal

        cfg = {
            "hidden_dim": hidden_dim,
            "node_emb_dim": node_emb_dim,
            "edge_emb_dim": edge_emb_dim,
            "use_goal": use_goal,
        }

        return {
            "state_dict": self.state_dict(),
            "config": cfg,
        }

    @classmethod
    def load_model(cls, ckpt: Dict):
        """
        Reconstructs a NewDistanceEstimator from a checkpoint dict
        produced by get_checkpoint().
        """
        cfg = ckpt["config"]
        # cfg contains the exact kwargs needed to reinstantiate
        model = cls(
            hidden_dim=cfg["hidden_dim"],
            use_goal=cfg["use_goal"],
            node_emb_dim=cfg["node_emb_dim"],
            edge_emb_dim=cfg["edge_emb_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])

        return model
