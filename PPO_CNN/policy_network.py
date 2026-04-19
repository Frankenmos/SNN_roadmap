import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from PPO_CNN.policy_input import (
    CURATED_FEATURE_UNIT_FIELDS,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_AVAILABLE_ACTION_DIM,
    META_LAST_ACTION_INDEX_DIM,
    META_PLAYER_FEATURE_DIM,
    PolicyInputBatch,
    SELECTION_FEATURE_DIM,
    SELECTION_UNIT_TYPE_INDEX,
    TOKEN_TYPE_GROUPS,
    UNIT_TYPE_FEATURE_INDEX,
)


class EntityEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        unit_type_vocab_size: int = 4096,
        unit_type_embed_dim: int | None = None,
    ):
        super().__init__()
        if feature_dim <= UNIT_TYPE_FEATURE_INDEX:
            raise ValueError(
                f"feature_dim must include unit_type, got {feature_dim}",
            )

        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.unit_type_index = int(UNIT_TYPE_FEATURE_INDEX)
        self.unit_type_embed_dim = int(
            unit_type_embed_dim or max(8, self.embed_dim // 4),
        )
        continuous_dim = self.feature_dim - 1 + self.unit_type_embed_dim

        self.unit_type_embedding = nn.Embedding(
            int(unit_type_vocab_size),
            self.unit_type_embed_dim,
            padding_idx=0,
        )
        self.pre_norm = nn.LayerNorm(continuous_dim)
        self.mlp = nn.Sequential(
            nn.Linear(continuous_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> torch.Tensor:
        unit_type_ids = entity_features[..., self.unit_type_index].round().long()
        unit_type_ids = unit_type_ids.clamp(
            0, self.unit_type_embedding.num_embeddings - 1,
        )
        unit_type_emb = self.unit_type_embedding(unit_type_ids)
        continuous = torch.cat(
            (
                entity_features[..., : self.unit_type_index],
                entity_features[..., self.unit_type_index + 1 :],
            ),
            dim=-1,
        )
        encoded = torch.cat((continuous, unit_type_emb), dim=-1)
        encoded = self.mlp(self.pre_norm(encoded))

        mask = entity_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return encoded * mask


class SelectionEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        unit_type_vocab_size: int = 4096,
        unit_type_embed_dim: int | None = None,
    ):
        super().__init__()
        if feature_dim <= SELECTION_UNIT_TYPE_INDEX:
            raise ValueError(
                f"feature_dim must include unit_type, got {feature_dim}",
            )

        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.unit_type_index = int(SELECTION_UNIT_TYPE_INDEX)
        self.unit_type_embed_dim = int(
            unit_type_embed_dim or max(8, self.embed_dim // 4),
        )
        continuous_dim = self.feature_dim - 1 + self.unit_type_embed_dim

        self.unit_type_embedding = nn.Embedding(
            int(unit_type_vocab_size),
            self.unit_type_embed_dim,
            padding_idx=0,
        )
        self.pre_norm = nn.LayerNorm(continuous_dim)
        self.mlp = nn.Sequential(
            nn.Linear(continuous_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(
        self,
        selection_features: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        unit_type_ids = selection_features[..., self.unit_type_index].round().long()
        unit_type_ids = unit_type_ids.clamp(
            0, self.unit_type_embedding.num_embeddings - 1,
        )
        unit_type_emb = self.unit_type_embedding(unit_type_ids)
        continuous = torch.cat(
            (
                selection_features[..., : self.unit_type_index],
                selection_features[..., self.unit_type_index + 1 :],
            ),
            dim=-1,
        )
        encoded = torch.cat((continuous, unit_type_emb), dim=-1)
        encoded = self.mlp(self.pre_norm(encoded))

        mask = selection_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return encoded * mask


class MetaEncoder(nn.Module):
    def __init__(
        self,
        meta_input_dim: int,
        embed_dim: int,
        player_dim: int = META_PLAYER_FEATURE_DIM,
        available_action_dim: int = META_AVAILABLE_ACTION_DIM,
        last_action_embed_dim: int | None = None,
        last_action_vocab_size: int = META_AVAILABLE_ACTION_DIM + 2,
    ):
        super().__init__()
        if meta_input_dim <= 0:
            raise ValueError("meta_input_dim must be positive")

        self.meta_input_dim = int(meta_input_dim)
        self.embed_dim = int(embed_dim)
        self.player_dim = int(player_dim)
        self.available_action_dim = int(available_action_dim)
        self.last_action_offset = self.player_dim + self.available_action_dim
        self.use_structured_meta = (
            self.meta_input_dim
            >= self.player_dim + self.available_action_dim + META_LAST_ACTION_INDEX_DIM
        )
        self.last_action_embed_dim = int(
            last_action_embed_dim or max(8, self.embed_dim // 4),
        )
        fused_input_dim = (
            self.player_dim
            + self.available_action_dim
            + self.last_action_embed_dim
            if self.use_structured_meta
            else self.meta_input_dim
        )

        self.last_action_embedding = nn.Embedding(
            int(last_action_vocab_size),
            self.last_action_embed_dim,
        )
        self.pre_norm = nn.LayerNorm(fused_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fused_input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, meta_vec: torch.Tensor) -> torch.Tensor:
        if self.use_structured_meta:
            player = meta_vec[..., : self.player_dim]
            available_actions = meta_vec[
                ...,
                self.player_dim : self.player_dim + self.available_action_dim
            ]
            last_action_ids = meta_vec[..., self.last_action_offset].round().long()
            last_action_ids = last_action_ids.clamp(
                0, self.last_action_embedding.num_embeddings - 1,
            )
            last_action_emb = self.last_action_embedding(last_action_ids)
            fused = torch.cat((player, available_actions, last_action_emb), dim=-1)
        else:
            fused = meta_vec

        return self.mlp(self.pre_norm(fused)).unsqueeze(1)


class SpikingSelfAttention(nn.Module):
    """
    Spikformer-style spike-driven self-attention.

    Computes spike(Q) @ spike(K)^T * scale @ spike(V), with Q/K/V
    encoded as binary spikes via LIF neurons.

    Stateless per forward() call. The Q/K/V LIF membranes are
    re-initialized every call, so this block does not carry state
    across env steps. Cross-step temporal memory lives in
    TokenTemporalSNN, which sits after this block.
    """

    def __init__(self, embed_dim=64, beta_qkv=0.5, spike_grad=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lif_q = snn.Leaky(
            beta=beta_qkv, spike_grad=spike_grad, learn_beta=True,
        )
        self.lif_k = snn.Leaky(
            beta=beta_qkv, spike_grad=spike_grad, learn_beta=True,
        )
        self.lif_v = snn.Leaky(
            beta=beta_qkv, spike_grad=spike_grad, learn_beta=True,
        )

    def forward(self, tokens, token_mask: torch.Tensor | None = None):
        tokens_normed = self.pre_norm(tokens)
        residual = tokens
        query_mask = None
        if token_mask is not None:
            query_mask = token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
            tokens_normed = tokens_normed * query_mask
            residual = residual * query_mask

        q_raw = self.W_q(tokens_normed)
        k_raw = self.W_k(tokens_normed)
        v_raw = self.W_v(tokens_normed)

        mem_q = self.lif_q.init_leaky()
        mem_k = self.lif_k.init_leaky()
        mem_v = self.lif_v.init_leaky()

        spike_q, _ = self.lif_q(q_raw, mem_q)
        spike_k, _ = self.lif_k(k_raw, mem_k)
        spike_v, _ = self.lif_v(v_raw, mem_v)

        if query_mask is not None:
            spike_q = spike_q * query_mask
            spike_k = spike_k * query_mask
            spike_v = spike_v * query_mask

        attn_logits = torch.bmm(spike_q, spike_k.transpose(1, 2)) * self.scale
        attn_logits = attn_logits.float()
        if token_mask is not None:
            key_mask = token_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(~key_mask, -1.0e4)
        attn = torch.softmax(attn_logits, dim=-1).to(dtype=tokens.dtype)
        out = torch.bmm(attn, spike_v)
        if query_mask is not None:
            out = out * query_mask
        return out + residual


class TokenTemporalSNN(nn.Module):
    """
    Cross-env-step temporal integrator over the token sequence.

    State: (syn, mem), each [B, N, D].
    """

    def __init__(self, alpha=0.8, beta=0.9, spike_grad=None):
        super().__init__()
        self.snn = snn.Synaptic(
            alpha=alpha,
            beta=beta,
            spike_grad=spike_grad,
            learn_alpha=True,
            learn_beta=True,
        )

    @staticmethod
    def init_state(batch_size, num_tokens, embed_dim, device, dtype):
        syn = torch.zeros(
            batch_size, num_tokens, embed_dim, device=device, dtype=dtype,
        )
        mem = torch.zeros(
            batch_size, num_tokens, embed_dim, device=device, dtype=dtype,
        )
        return (syn, mem)

    def forward(self, tokens, syn, mem):
        spk, syn, mem = self.snn(tokens, syn, mem)
        return spk, syn, mem


class PolicyNetwork(nn.Module):
    """
    Spiking Transformer policy with token-level temporal state.

    Pipeline:
      raw obs
        -> conv backbone
        -> pooled spatial tokens
        -> spiking self-attention
        -> token temporal SNN
        -> shared readout
        -> action/value heads
    """

    def __init__(
        self,
        spatial_input_shape,
        vector_input_dim,
        action_dim,
        num_steps=1,
        screen_size=84,
        token_snn_alpha=0.8,
        token_snn_beta=0.9,
        attention_embed_dim=64,
        attention_pool_size=7,
        attention_beta=0.5,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = max(1, int(num_steps))
        self.screen_size = int(screen_size)
        self._pool_size = max(1, int(attention_pool_size))
        self._embed_dim = max(1, int(attention_embed_dim))
        self._spatial_tokens = self._pool_size * self._pool_size
        self._entity_start = self._spatial_tokens
        self._entity_end = self._entity_start + MAX_ENTITY_TOKENS
        self._selection_start = self._entity_end
        self._selection_end = self._selection_start + MAX_SELECTION_TOKENS
        self._meta_start = self._selection_end
        self._num_tokens = (
            self._spatial_tokens + MAX_ENTITY_TOKENS + MAX_SELECTION_TOKENS + 1
        )
        self._carry_entity_state = False
        self._meta_input_dim = int(vector_input_dim)
        self._config = {
            "num_steps": self.num_steps,
            "screen_size": self.screen_size,
            "token_snn_alpha": float(token_snn_alpha),
            "token_snn_beta": float(token_snn_beta),
            "attention_embed_dim": self._embed_dim,
            "attention_pool_size": self._pool_size,
            "attention_beta": float(attention_beta),
            "meta_input_dim": self._meta_input_dim,
            "carry_entity_state": bool(self._carry_entity_state),
        }
        spike_grad = surrogate.fast_sigmoid()

        c_in = spatial_input_shape[0]
        self.conv1 = nn.Conv2d(c_in, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(
            32, self._embed_dim, kernel_size=3, stride=1, padding=1,
        )

        self.token_pool = nn.AdaptiveAvgPool2d(
            (self._pool_size, self._pool_size),
        )
        self.entity_encoder = EntityEncoder(
            feature_dim=len(CURATED_FEATURE_UNIT_FIELDS),
            embed_dim=self._embed_dim,
        )
        self.selection_encoder = SelectionEncoder(
            feature_dim=SELECTION_FEATURE_DIM,
            embed_dim=self._embed_dim,
        )
        self.meta_encoder = MetaEncoder(
            meta_input_dim=self._meta_input_dim,
            embed_dim=self._embed_dim,
        )
        self.token_type_embedding = nn.Embedding(
            TOKEN_TYPE_GROUPS,
            self._embed_dim,
        )
        self.attention = SpikingSelfAttention(
            embed_dim=self._embed_dim,
            beta_qkv=attention_beta,
            spike_grad=spike_grad,
        )
        self.token_snn = TokenTemporalSNN(
            alpha=token_snn_alpha,
            beta=token_snn_beta,
            spike_grad=spike_grad,
        )

        fc_input_dim = self._num_tokens * self._embed_dim
        self.combined_norm = nn.LayerNorm(fc_input_dim)
        self.shared_fc1 = nn.Linear(fc_input_dim, 128)
        self.shared_fc2 = nn.Linear(128, 64)

        self.actor_fc = nn.Linear(64, action_dim)
        self.critic_fc = nn.Linear(64, 1)
        self.move_x_fc = nn.Linear(64, self.screen_size)
        self.move_y_fc = nn.Linear(64, self.screen_size)

        self.use_amp = torch.cuda.is_available()
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.to(self.device)

    def resolved_config(self):
        return {
            **self._config,
            "num_tokens": self._num_tokens,
            "spatial_tokens": self._spatial_tokens,
            "amp_enabled": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype),
            "device": str(self.device),
        }

    def _zero_entity_state(
        self,
        syn_tok: torch.Tensor,
        mem_tok: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._carry_entity_state:
            return syn_tok, mem_tok
        syn_tok = syn_tok.clone()
        mem_tok = mem_tok.clone()
        syn_tok[:, self._entity_start : self._entity_end, :] = 0.0
        mem_tok[:, self._entity_start : self._entity_end, :] = 0.0
        return syn_tok, mem_tok

    def init_concrete_state(self, batch_size=1, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return TokenTemporalSNN.init_state(
            batch_size, self._num_tokens, self._embed_dim, device, dtype,
        )

    def _add_token_type(
        self,
        tokens: torch.Tensor,
        token_type_index: int,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_emb = self.token_type_embedding.weight[token_type_index].view(1, 1, -1)
        typed = tokens + type_emb.to(dtype=tokens.dtype, device=tokens.device)
        if token_mask is not None:
            typed = typed * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return typed

    def forward(self, batch: PolicyInputBatch):
        if not isinstance(batch, PolicyInputBatch):
            raise TypeError(
                f"PolicyNetwork.forward expects PolicyInputBatch, got {type(batch)!r}",
            )

        spatial_input = batch.spatial_obs
        token_state = batch.state_in
        if token_state is None:
            syn_tok, mem_tok = self.init_concrete_state(
                batch_size=spatial_input.size(0),
                device=spatial_input.device,
                dtype=spatial_input.dtype,
            )
        else:
            syn_tok, mem_tok = token_state
        syn_tok, mem_tok = self._zero_entity_state(syn_tok, mem_tok)

        x = F.relu(self.conv1(spatial_input))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        spatial_tokens = self.token_pool(x)
        spatial_tokens = spatial_tokens.flatten(2).transpose(1, 2)
        batch_size = spatial_tokens.size(0)
        device = spatial_tokens.device
        spatial_mask = torch.ones(
            batch_size,
            self._spatial_tokens,
            dtype=torch.bool,
            device=device,
        )
        entity_tokens = self.entity_encoder(
            batch.entity_features,
            batch.entity_mask,
        )
        selection_tokens = self.selection_encoder(
            batch.selection_features,
            batch.selection_mask,
        )
        meta_tokens = self.meta_encoder(batch.meta_vec)
        meta_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        tokens = torch.cat(
            (
                self._add_token_type(spatial_tokens, 0, spatial_mask),
                self._add_token_type(entity_tokens, 1, batch.entity_mask),
                self._add_token_type(selection_tokens, 2, batch.selection_mask),
                self._add_token_type(meta_tokens, 3, meta_mask),
            ),
            dim=1,
        )
        token_mask = torch.cat(
            (
                spatial_mask,
                batch.entity_mask,
                batch.selection_mask,
                meta_mask,
            ),
            dim=1,
        )
        attended = self.attention(tokens, token_mask=token_mask)

        spike_rec = []
        token_mask_f = token_mask.unsqueeze(-1).to(dtype=attended.dtype)
        syn_tok = syn_tok * token_mask_f
        mem_tok = mem_tok * token_mask_f
        for _ in range(self.num_steps):
            spk_tok, syn_tok, mem_tok = self.token_snn(attended, syn_tok, mem_tok)
            spk_tok = spk_tok * token_mask_f
            syn_tok = syn_tok * token_mask_f
            mem_tok = mem_tok * token_mask_f
            spike_rec.append(spk_tok)
        syn_tok, mem_tok = self._zero_entity_state(syn_tok, mem_tok)

        aggregated = torch.stack(spike_rec, dim=0).sum(dim=0)
        combined = self.combined_norm(aggregated.flatten(start_dim=1))

        x = F.relu(self.shared_fc1(combined))
        x = F.relu(self.shared_fc2(x))

        action_logits = self.actor_fc(x)
        move_x_logits = self.move_x_fc(x)
        move_y_logits = self.move_y_fc(x)
        state_value = self.critic_fc(x).squeeze(-1)
        next_state = (syn_tok, mem_tok)
        return action_logits, move_x_logits, move_y_logits, state_value, next_state
