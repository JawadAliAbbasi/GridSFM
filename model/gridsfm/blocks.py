"""Building blocks and constants for the GridSFM backbone."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from .schema import BUS_TYPE_IDX
from .signed_incidence import SignedIncidenceConv

EdgeType = Tuple[str, str, str]
_G_CTX_DIM = 8

DEFAULT_INPUT_DIMS = {
    'bus':       4 + 5 + 3 + 4,
    'generator': 11,
    'load':      2,
    'shunt':     2,
    'branch_ac': 9 + 4,
    'branch_tr': 11 + 4,
    'cycle':     4,
}

# CoED (continuous edge directions) applies to the branch-endpoint incidence
# only.  Those four relations carry the branch orientation as a hard +/-1 in
# edge_attr[:, 0] (+1 for the "from" endpoint, -1 for the "to" endpoint; see
# cycle_basis._promote_branch_nodes), which is exactly the discrete direction
# CoED exists to relax.  The remaining signed relations (cycle <-> branch) carry
# cycle-basis orientation, which is not an edge direction, so they are excluded.
_COED_TYPES = {
    ('bus',       'endpoint_of', 'branch_ac'),
    ('branch_ac', 'endpoint_of', 'bus'),
    ('bus',       'endpoint_of', 'branch_tr'),
    ('branch_tr', 'endpoint_of', 'bus'),
}

_SIGNED_INCIDENCE_TYPES = {
    ('bus',         'endpoint_of', 'branch_ac'),
    ('branch_ac',   'endpoint_of', 'bus'),
    ('bus',         'endpoint_of', 'branch_tr'),
    ('branch_tr',   'endpoint_of', 'bus'),
    ('cycle',       'in_cycle',    'branch_ac'),
    ('branch_ac',   'in_cycle',    'cycle'),
    ('cycle',       'in_cycle',    'branch_tr'),
    ('branch_tr',   'in_cycle',    'cycle'),
}


def _nan_to_zero(x: Tensor, sub_value: float = 0.0) -> Tensor:
    return torch.nan_to_num(x, nan=sub_value, posinf=sub_value, neginf=sub_value)


def _leaky_clip(u: Tensor, lo: Tensor, hi: Tensor, leak: float = 0.0) -> Tensor:
    return torch.clamp(u, lo, hi) + leak * (
        (u - hi).clamp_min(0.0) - (lo - u).clamp_min(0.0)
    )


def _scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if src.size(0) == 0:
        return torch.zeros(dim_size, src.size(-1) if src.dim() > 1 else 1,
                           dtype=src.dtype, device=src.device)
    if src.dim() == 1:
        src = src.unsqueeze(-1)
    out = torch.zeros(dim_size, src.size(-1), dtype=src.dtype, device=src.device)
    cnt = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(-1)), src)
    cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    out = out / cnt.clamp_min(1.0).unsqueeze(-1)
    return out


def _slack_or_mean_anchor(data: HeteroData, x_bus: Tensor, n_graphs: int) -> Tensor:
    bus_batch = data['bus'].batch
    bus_type = data['bus'].x[:, BUS_TYPE_IDX].long()
    is_slack = (bus_type == 3).to(torch.float32)
    dev = x_bus.device

    slack_cnt = torch.zeros(n_graphs, device=dev, dtype=torch.float32)
    slack_cnt.scatter_add_(0, bus_batch, is_slack)
    all_cnt = torch.zeros(n_graphs, device=dev, dtype=torch.float32)
    all_cnt.scatter_add_(0, bus_batch, torch.ones_like(bus_batch, dtype=torch.float32))

    x32 = x_bus.to(torch.float32)
    slack_sum = torch.zeros(n_graphs, device=dev, dtype=torch.float32)
    slack_sum.scatter_add_(0, bus_batch, x32 * is_slack)
    all_sum = torch.zeros(n_graphs, device=dev, dtype=torch.float32)
    all_sum.scatter_add_(0, bus_batch, x32)

    anchor = torch.where(
        slack_cnt > 0,
        slack_sum / slack_cnt.clamp_min(1.0),
        all_sum / all_cnt.clamp_min(1.0),
    ).detach()
    return x_bus - anchor[bus_batch].to(x_bus.dtype)


class _SAGEConvWithEdgeAttr(SAGEConv):
    """SAGEConv that accepts and ignores an `edge_attr` kwarg."""
    def _coed_edge_weight(self, branch_nt: str, num_edges: int, device, dtype) -> Tensor:
        """Per-edge multiplier for the signed branch-endpoint incidence.

        Each branch node receives one message per endpoint bus, scaled by a hard
        sign: +1 from its "from" bus, -1 from its "to" bus.  CoED relaxes that
        hard direction into a learnable angle, weighting the two endpoints by

            w_from = 2 cos^2(theta),   w_to = 2 sin^2(theta)

        which sum to 2, so at theta = pi/4 both are exactly 1 and the layer
        reduces to the unmodified signed incidence.  The factor 2 is what makes
        that an exact no-op; without it every message would be halved at init.

        Layout.  Within one graph the endpoint edges are contiguous,
        ``[all from-endpoints, all to-endpoints]``.  PyG batches by
        concatenating whole graphs, so for ``g`` graphs the per-graph pattern
        REPEATS rather than the two halves staying globally contiguous:
        ``[g1_from, g1_to, g2_from, g2_to, ...]``.  Hence ``.repeat(g)`` of the
        single-graph vector, not ``repeat`` of each half.  Getting this backwards
        binds theta to the wrong endpoints and raises nothing.
        """
        theta_raw = self.coed_theta_raw[branch_nt]
        n_branch = theta_raw.numel()
        if n_branch == 0:
            return torch.ones(num_edges, 1, device=device, dtype=dtype)
        if num_edges % (2 * n_branch) != 0:
            raise ValueError(
                f"CoED expects {2 * n_branch} endpoint edges per graph for "
                f"{branch_nt} ({n_branch} branches x 2 endpoints), but the batch "
                f"has {num_edges}, which is not a whole multiple. Either the "
                f"topology is not fixed across samples or coed_branch_counts is "
                f"wrong for this grid."
            )
        n_graphs = num_edges // (2 * n_branch)

        # Written as 1 +/- cos(2*theta) rather than 2*cos^2 / 2*sin^2 -- the same
        # quantity, but exact at the initialisation point.  Evaluating
        # 2*cos(pi/4)**2 gives 0.99999994 in float32 and 1.0000000000000002 in
        # float64, whereas cos(2*theta) at theta = pi/4 is ~6e-17 and 1 +/- it
        # casts to exactly 1.0 in float32 -- the dtype the incidence sign is
        # actually built in (cycle_basis makes side_sign float32).  So an
        # untrained CoED run is bit-identical to the baseline, which is how we
        # verify theta is wired up rather than silently inert.  In float64 the
        # "to" weight lands one ulp low (1 - 1.1e-16); harmless, but it means the
        # no-op is exact only at float32.  Computed in double, then cast.
        theta = (torch.pi / 2.0) * torch.sigmoid(theta_raw.double())
        cos2t = torch.cos(2.0 * theta)
        w_from = 1.0 + cos2t
        w_to = 1.0 - cos2t
        w = torch.cat([w_from, w_to]).repeat(n_graphs)
        return w.to(device=device, dtype=dtype).view(-1, 1)

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return super().forward(x, edge_index, **kwargs)


def _patch_conv_mask_no_edge_graph(conv: nn.Module) -> nn.Module:
    orig_forward = conv.forward

    def masked_forward(x, edge_index, *args, **kwargs):
        # GridBlock.forward always threads batch_dst + n_graphs through
        # HeteroConv, so kwargs.pop is guaranteed to find them.
        batch_dst = kwargs.pop("batch_dst")
        n_graphs = kwargs.pop("n_graphs")
        out = orig_forward(x, edge_index, *args, **kwargs)
        if edge_index.numel() == 0:
            return torch.zeros_like(out)
        b_dst = batch_dst[1] if isinstance(batch_dst, (tuple, list)) else batch_dst
        if b_dst is None or b_dst.numel() == 0:
            return out
        graph_has_edge = torch.zeros(n_graphs, dtype=torch.bool, device=out.device)
        graph_has_edge[b_dst[edge_index[1]]] = True
        mask = graph_has_edge[b_dst]
        return out * mask.to(out.dtype).unsqueeze(-1)

    conv.forward = masked_forward
    return conv


class LinearSelfAttention(nn.Module):
    """Per-graph, per-type linear-time self-attention (Performer-style)."""
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.normal_(self.W_o.weight, mean=0.0, std=0.02)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor, batch: Tensor, n_graphs: int) -> Tensor:
        if h.size(0) == 0:
            return h
        N = h.size(0)
        H = self.num_heads
        Dk = self.d_k

        Q = self.W_q(h).view(N, H, Dk)
        K = self.W_k(h).view(N, H, Dk)
        V = self.W_v(h).view(N, H, Dk)

        Qp = torch.nn.functional.elu(Q) + 1.0
        Kp = torch.nn.functional.elu(K) + 1.0

        dev = h.device
        KV = torch.zeros(n_graphs, H, Dk, Dk, dtype=h.dtype, device=dev)
        K_sum = torch.zeros(n_graphs, H, Dk, dtype=h.dtype, device=dev)

        outer = Kp.unsqueeze(-1) * V.unsqueeze(-2)
        b_idx = batch.long()
        b_exp = b_idx.view(N, 1, 1, 1).expand(-1, H, Dk, Dk)
        KV.scatter_add_(0, b_exp, outer)
        b_exp_k = b_idx.view(N, 1, 1).expand(-1, H, Dk)
        K_sum.scatter_add_(0, b_exp_k, Kp)

        KV_at_node   = KV[b_idx]
        Ksum_at_node = K_sum[b_idx]
        num   = (Qp.unsqueeze(-2) @ KV_at_node).squeeze(-2)
        denom = (Qp * Ksum_at_node).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn  = num / denom

        out = attn.contiguous().view(N, H * Dk)
        out = self.W_o(out)
        return self.drop(out)


class GridBlock(nn.Module):
    """Transformer-style block over a heterogeneous PyG graph."""
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[EdgeType],
        hidden_dim: int,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        coed_branch_counts: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        # --- CoED: one learnable angle per undirected branch, per block -------
        # theta = (pi/2) * sigmoid(theta_raw) keeps the angle in [0, pi/2] by
        # construction, so no clamping hook is needed, and theta_raw = 0 gives
        # exactly theta = pi/4.  There the two endpoint weights are both 1 (see
        # _coed_edge_weight), so an untrained CoED model is bit-identical to the
        # baseline -- which is the check that CoED is wired up rather than inert.
        #
        # The parameter lives on the block, so each block learns its own angles.
        # A single shared theta would apply the same directional prior at every
        # block and compound with depth.
        self.coed_branch_counts = dict(coed_branch_counts) if coed_branch_counts else None
        if self.coed_branch_counts:
            self.coed_theta_raw = nn.ParameterDict({
                nt: nn.Parameter(torch.zeros(int(n)))
                for nt, n in self.coed_branch_counts.items()
            })
        else:
            self.coed_theta_raw = None

        self.attn_ln  = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
        self.attn     = nn.ModuleDict({
            nt: LinearSelfAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
            for nt in node_types
        })

        self.gnn_ln   = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
        conv_dict = {}
        for et in edge_types:
            if et in _SIGNED_INCIDENCE_TYPES:
                conv = SignedIncidenceConv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=hidden_dim,
                )
            else:
                conv = _SAGEConvWithEdgeAttr(
                    (hidden_dim, hidden_dim), hidden_dim,
                    aggr='mean', root_weight=True,
                )
            conv_dict[et] = _patch_conv_mask_no_edge_graph(conv)
        self.gnn = HeteroConv(conv_dict, aggr='sum')

        self.ffn_ln   = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
        self.ffn      = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(hidden_dim, ffn_mult * hidden_dim),
                nn.GELU(),
                nn.Linear(ffn_mult * hidden_dim, hidden_dim),
            ) for nt in node_types
        })
        for nt in node_types:
            nn.init.normal_(self.ffn[nt][-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.ffn[nt][-1].bias)

        self.node_types = list(node_types)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        batch_dict: Dict[str, Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]],
        n_graphs: int,
    ) -> Dict[str, Tensor]:
        x_dict = {
            nt: (x + self.attn[nt](self.attn_ln[nt](x), batch_dict[nt], n_graphs)
                 if nt in self.attn else x)
            for nt, x in x_dict.items()
        }

        x_pre = {nt: self.gnn_ln[nt](x) for nt, x in x_dict.items() if nt in self.gnn_ln}
        if edge_attr_dict is None:
            edge_attr_dict = {}
        if self.coed_theta_raw is not None:
            # Fold the CoED weight into the sign column itself.  SignedIncidenceConv
            # reads only edge_attr[:, 0:1], and HeteroConv already routes
            # edge_attr_dict per relation, so no conv or propagate signature has
            # to change.  Copy first: the caller's dict is reused by every block.
            edge_attr_dict = dict(edge_attr_dict)
            for et in _COED_TYPES:
                ea = edge_attr_dict.get(et)
                if ea is None or et not in edge_index_dict:
                    continue
                branch_nt = et[2] if et[0] == 'bus' else et[0]
                if branch_nt not in self.coed_theta_raw:
                    continue
                edge_attr_dict[et] = ea * self._coed_edge_weight(
                    branch_nt, ea.size(0), ea.device, ea.dtype,
                )
        n_graphs_dict = {et: n_graphs for et in edge_index_dict}
        x_msg = self.gnn(x_pre, edge_index_dict,
                         edge_attr_dict=edge_attr_dict,
                         batch_dst_dict=batch_dict,
                         n_graphs_dict=n_graphs_dict)
        x_dict = {
            nt: (x + x_msg[nt]) if nt in x_msg else x
            for nt, x in x_dict.items()
        }

        x_dict = {
            nt: (x + self.ffn[nt](self.ffn_ln[nt](x))
                 if nt in self.ffn else x)
            for nt, x in x_dict.items()
        }
        return x_dict


class FusionLayer(nn.Module):
    """Fuse per-type embeddings into (h_grid_bus, h_grid_gen, h_grid_global)."""
    def __init__(self, hidden_dim: int, signed_fusion: bool):
        super().__init__()
        d = hidden_dim
        self.signed_fusion = bool(signed_fusion)
        self.W_b_self    = nn.Linear(d, d, bias=False)
        self.W_b_from_ac = nn.Linear(d, d, bias=False)
        self.W_b_from_tr = nn.Linear(d, d, bias=False)
        self.W_b_from_g  = nn.Linear(d, d, bias=False)
        self.W_b_from_l  = nn.Linear(d, d, bias=False)
        self.W_b_from_s  = nn.Linear(d, d, bias=False)
        self.W_b_from_c  = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W_b_self.weight)
        for w in [self.W_b_from_ac, self.W_b_from_tr, self.W_b_from_g,
                  self.W_b_from_l, self.W_b_from_s, self.W_b_from_c]:
            nn.init.normal_(w.weight, mean=0.0, std=0.02)
        self.bus_ln = nn.LayerNorm(d)

        self.W_g_self    = nn.Linear(d, d, bias=False)
        self.W_g_from_b  = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W_g_self.weight)
        nn.init.normal_(self.W_g_from_b.weight, mean=0.0, std=0.02)
        self.gen_ln = nn.LayerNorm(d)

        self.W_global = nn.Linear(8 * d, d)
        self.global_ln = nn.LayerNorm(d)

    def forward(
        self,
        h: Dict[str, Tensor],
        data: HeteroData,
        n_graphs: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        h_bus = h['bus']
        n_bus = h_bus.size(0)
        d = h_bus.size(-1)
        dev = h_bus.device

        def _agg_to_bus(h_src: Tensor, bus_index: Tensor,
                        signs: Optional[Tensor] = None) -> Tensor:
            if h_src is None or h_src.size(0) == 0 or bus_index.numel() == 0:
                return torch.zeros(n_bus, d, dtype=h_bus.dtype, device=dev)
            msgs = h_src[bus_index[0]]
            if signs is not None:
                if signs.dim() == 1:
                    signs = signs.unsqueeze(-1)
                msgs = msgs * signs
            return _scatter_mean(msgs, bus_index[1], n_bus)

        agg_ac = torch.zeros(n_bus, d, dtype=h_bus.dtype, device=dev)
        if 'branch_ac' in h and ('branch_ac', 'endpoint_of', 'bus') in data.edge_types:
            ei = data[('branch_ac', 'endpoint_of', 'bus')].edge_index
            ea = data[('branch_ac', 'endpoint_of', 'bus')].edge_attr
            signs_ac = (ea[:, 0] if (self.signed_fusion and ea is not None
                                     and ea.size(1) >= 1)
                        else None)
            agg_ac = _agg_to_bus(h['branch_ac'], ei, signs_ac)
        agg_tr = torch.zeros(n_bus, d, dtype=h_bus.dtype, device=dev)
        if 'branch_tr' in h and ('branch_tr', 'endpoint_of', 'bus') in data.edge_types:
            ei = data[('branch_tr', 'endpoint_of', 'bus')].edge_index
            ea = data[('branch_tr', 'endpoint_of', 'bus')].edge_attr
            signs_tr = (ea[:, 0] if (self.signed_fusion and ea is not None
                                     and ea.size(1) >= 1)
                        else None)
            agg_tr = _agg_to_bus(h['branch_tr'], ei, signs_tr)
        agg_g = torch.zeros(n_bus, d, dtype=h_bus.dtype, device=dev)
        if 'generator' in h and ('generator', 'generator_link', 'bus') in data.edge_types:
            agg_g = _agg_to_bus(h['generator'], data[('generator', 'generator_link', 'bus')].edge_index)
        agg_l = torch.zeros(n_bus, d, dtype=h_bus.dtype, device=dev)
        if 'load' in h and ('load', 'load_link', 'bus') in data.edge_types:
            agg_l = _agg_to_bus(h['load'], data[('load', 'load_link', 'bus')].edge_index)
        agg_s = torch.zeros(n_bus, d, dtype=h_bus.dtype, device=dev)
        if 'shunt' in h and ('shunt', 'shunt_link', 'bus') in data.edge_types:
            agg_s = _agg_to_bus(h['shunt'], data[('shunt', 'shunt_link', 'bus')].edge_index)

        agg_c = torch.zeros(n_bus, d, dtype=h_bus.dtype, device=dev)
        if 'cycle' in h and h['cycle'].size(0) > 0:
            for et_branch in ('branch_ac', 'branch_tr'):
                bipartite = ('cycle', 'in_cycle', et_branch)
                bus_endpoint = (et_branch, 'endpoint_of', 'bus')
                if bipartite not in data.edge_types or bus_endpoint not in data.edge_types:
                    continue
                ce = data[bipartite].edge_index
                ce_attr = data[bipartite].edge_attr
                if ce.numel() == 0 or et_branch not in h:
                    continue
                n_branches = h[et_branch].size(0)
                cyc_msgs = h['cycle'][ce[0]]
                if (self.signed_fusion and ce_attr is not None
                        and ce_attr.size(1) >= 1):
                    cyc_msgs = cyc_msgs * ce_attr[:, 0:1]
                cycle_to_branch = _scatter_mean(cyc_msgs, ce[1], n_branches)
                ep_ea = data[bus_endpoint].edge_attr
                signs_ep = (ep_ea[:, 0] if (self.signed_fusion and ep_ea is not None
                                            and ep_ea.size(1) >= 1)
                            else None)
                agg_c_partial = _agg_to_bus(cycle_to_branch,
                                            data[bus_endpoint].edge_index,
                                            signs=signs_ep)
                agg_c = agg_c + agg_c_partial

        h_grid_bus = self.bus_ln(
            self.W_b_self(h_bus)
          + self.W_b_from_ac(agg_ac)
          + self.W_b_from_tr(agg_tr)
          + self.W_b_from_g (agg_g)
          + self.W_b_from_l (agg_l)
          + self.W_b_from_s (agg_s)
          + self.W_b_from_c (agg_c)
        )

        h_grid_gen = h['generator']
        if ('generator', 'generator_link', 'bus') in data.edge_types and h['generator'].size(0) > 0:
            gl = data[('generator', 'generator_link', 'bus')].edge_index
            bus_for_gen = h_grid_bus[gl[1]]
            h_grid_gen = self.gen_ln(self.W_g_self(h['generator']) + self.W_g_from_b(bus_for_gen))
        else:
            h_grid_gen = self.gen_ln(self.W_g_self(h_grid_gen))

        bus_batch = data['bus'].batch

        def _pool_per_type(h_src, batch_src):
            """Return (mean, max) per graph; both [n_graphs, d]. Empty-group
            max rows are clamped to 0 to match the all-zero-input contract."""
            if h_src.size(0) == 0:
                z = torch.zeros(n_graphs, d, dtype=h_bus.dtype, device=dev)
                return z, z
            mean = _scatter_mean(h_src, batch_src, n_graphs)
            mx = torch.full((n_graphs, d), float('-inf'),
                            dtype=h_src.dtype, device=dev).scatter_reduce(
                                0, batch_src.unsqueeze(-1).expand(-1, d),
                                h_src, reduce='amax', include_self=True)
            mx = mx.masked_fill(mx == float('-inf'), 0.0)
            return mean, mx

        empty_h = torch.zeros(0, d, device=dev)
        empty_b = torch.zeros(0, dtype=torch.long, device=dev)

        def _hb(nt):
            present = nt in data.node_types and data[nt].x.size(0) > 0
            return ((h.get(nt, empty_h), data[nt].batch) if present
                    else (empty_h, empty_b))

        # Per-type (mean, max) pooled to a single d-vector each; concat
        # interleaved as 8*d for W_global. Layout drives the cross-version
        # adapter; keep mean/max paired per type.
        mean_bus, max_bus = _pool_per_type(h_grid_bus, bus_batch)
        mean_ac,  max_ac  = _pool_per_type(*_hb('branch_ac'))
        mean_tr,  max_tr  = _pool_per_type(*_hb('branch_tr'))
        mean_cyc, max_cyc = _pool_per_type(*_hb('cycle'))
        h_grid_global = self.global_ln(self.W_global(
            torch.cat([mean_bus, max_bus, mean_ac, max_ac,
                       mean_tr, max_tr, mean_cyc, max_cyc], dim=-1)
        ))

        return h_grid_bus, h_grid_gen, h_grid_global
