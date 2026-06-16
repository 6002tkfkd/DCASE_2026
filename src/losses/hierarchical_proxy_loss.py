import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalProxyLoss(nn.Module):
    """Proxy loss with separate parent and child proxy spaces."""

    requires_hierarchical_labels = True

    def __init__(
        self,
        embedding_dim=64,
        num_parents=5,
        num_children=23,
        temperature=0.07,
        alpha=0.4,
        beta=0.3,
        gamma=0.15,
        delta=0.05,
        sibling_margin=0.4,
        parent_margin=0.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_parents = num_parents
        self.num_children = num_children
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sibling_margin = sibling_margin
        self.parent_margin = parent_margin

        self.parent_proxies = nn.Parameter(torch.randn(num_parents, embedding_dim))
        self.child_proxies = nn.Parameter(torch.randn(num_children, embedding_dim))
        nn.init.xavier_uniform_(self.parent_proxies)
        nn.init.xavier_uniform_(self.child_proxies)

    def forward(self, z, parent_labels, child_labels, parent_of_child):
        parent_proxies = F.normalize(self.parent_proxies, dim=1)
        child_proxies = F.normalize(self.child_proxies, dim=1)
        z = F.normalize(z, dim=1)

        child_logits = torch.matmul(z, child_proxies.T) / self.temperature
        loss_child_cls = F.cross_entropy(child_logits, child_labels)

        parent_logits = torch.matmul(z, parent_proxies.T) / self.temperature
        loss_parent_cls = F.cross_entropy(parent_logits, parent_labels)

        child_parent_proxy = parent_proxies[parent_of_child]
        child_to_parent_sim = torch.sum(child_proxies * child_parent_proxy, dim=1)
        loss_child_to_parent = torch.mean(1.0 - child_to_parent_sim)

        child_sim = torch.matmul(child_proxies, child_proxies.T)
        same_parent = parent_of_child.unsqueeze(0) == parent_of_child.unsqueeze(1)
        not_self = ~torch.eye(self.num_children, dtype=torch.bool, device=z.device)
        sibling_sim = child_sim[same_parent & not_self]
        if sibling_sim.numel() > 0:
            loss_sibling_sep = F.relu(sibling_sim - self.sibling_margin).mean()
        else:
            loss_sibling_sep = torch.tensor(0.0, device=z.device)

        parent_sim = torch.matmul(parent_proxies, parent_proxies.T)
        parent_not_self = ~torch.eye(self.num_parents, dtype=torch.bool, device=z.device)
        parent_pair_sim = parent_sim[parent_not_self]
        if parent_pair_sim.numel() > 0:
            loss_parent_sep = F.relu(parent_pair_sim - self.parent_margin).mean()
        else:
            loss_parent_sep = torch.tensor(0.0, device=z.device)

        total_loss = (
            loss_child_cls
            + self.alpha * loss_parent_cls
            + self.beta * loss_child_to_parent
            + self.gamma * loss_sibling_sep
            + self.delta * loss_parent_sep
        )

        loss_dict = {
            "child_cls": loss_child_cls,
            "parent_cls": loss_parent_cls,
            "child_to_parent": loss_child_to_parent,
            "sibling_sep": loss_sibling_sep,
            "parent_sep": loss_parent_sep,
        }
        return total_loss, loss_dict
