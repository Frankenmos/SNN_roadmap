import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from agent_core.policy_protocol import (
    AGENT_LAST_ACTION_OFFSET,
    AGENT_LAST_ACTION_DIM,
    BRIDGE_ACTION_VOCAB_SIZE,
    CURATED_FEATURE_UNIT_FIELDS,
    MAX_ENTITY_TOKENS,
    MAX_SELECTION_TOKENS,
    META_AVAILABLE_ACTION_OFFSET,
    META_AVAILABLE_ACTION_DIM,
    META_LAST_ACTION_INDEX_OFFSET,
    META_LAST_ACTION_INDEX_DIM,
    META_PLAYER_FEATURE_DIM,
    PolicyInputBatch,
    POLICY_ACTION_NO_OP,
    SELECTION_FEATURE_DIM,
    SELECTION_UNIT_TYPE_INDEX,
    TOKEN_TYPE_GROUPS,
    UNKNOWN_LAST_ACTION_INDEX,
    UNIT_TYPE_FEATURE_INDEX,
)
from agent_core.target_heads import (
    CoarseToFineTargetHead,
    FactorizedXYTargetHead,
    TargetEval,
    TargetHeadState,
    TargetSample,
    TokenPointerTargetHead,
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
        last_action_vocab_size: int = UNKNOWN_LAST_ACTION_INDEX + 1,
        bridge_action_embed_dim: int | None = None,
        bridge_action_vocab_size: int = BRIDGE_ACTION_VOCAB_SIZE,
    ):
        super().__init__()
        if meta_input_dim <= 0:
            raise ValueError("meta_input_dim must be positive")

        self.meta_input_dim = int(meta_input_dim)
        self.embed_dim = int(embed_dim)
        self.player_dim = int(player_dim)
        self.available_action_dim = int(available_action_dim)
        self.last_action_offset = int(META_LAST_ACTION_INDEX_OFFSET)
        self.agent_last_action_offset = int(AGENT_LAST_ACTION_OFFSET)
        self.agent_last_action_dim = int(AGENT_LAST_ACTION_DIM)
        self.use_structured_meta = self.meta_input_dim >= (
            self.player_dim + self.available_action_dim + META_LAST_ACTION_INDEX_DIM
        )
        self.use_agent_last_action = (
            self.meta_input_dim >= self.agent_last_action_offset + self.agent_last_action_dim
        )
        self.last_action_embed_dim = int(
            last_action_embed_dim or max(8, self.embed_dim // 4),
        )
        self.bridge_action_embed_dim = int(
            bridge_action_embed_dim or max(4, self.embed_dim // 8),
        )
        fused_input_dim = (
            self.player_dim
            + self.available_action_dim
            + self.last_action_embed_dim
            + (
                self.bridge_action_embed_dim + (self.agent_last_action_dim - 1)
                if self.use_agent_last_action
                else 0
            )
            if self.use_structured_meta
            else self.meta_input_dim
        )

        self.last_action_embedding = nn.Embedding(
            int(last_action_vocab_size),
            self.last_action_embed_dim,
        )
        self.bridge_action_embedding = nn.Embedding(
            int(bridge_action_vocab_size),
            self.bridge_action_embed_dim,
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
                META_AVAILABLE_ACTION_OFFSET : META_AVAILABLE_ACTION_OFFSET
                + self.available_action_dim
            ]
            last_action_ids = meta_vec[..., self.last_action_offset].round().long()
            last_action_ids = last_action_ids.clamp(
                0, self.last_action_embedding.num_embeddings - 1,
            )
            last_action_emb = self.last_action_embedding(last_action_ids)
            fused_parts = [player, available_actions, last_action_emb]
            if self.use_agent_last_action:
                agent_token = meta_vec[
                    ...,
                    self.agent_last_action_offset : self.agent_last_action_offset
                    + self.agent_last_action_dim
                ]
                agent_type_ids = agent_token[..., 0].round().long().clamp(
                    0, self.bridge_action_embedding.num_embeddings - 1,
                )
                agent_type_emb = self.bridge_action_embedding(agent_type_ids)
                fused_parts.append(agent_type_emb)
                fused_parts.append(agent_token[..., 1:])
            fused = torch.cat(fused_parts, dim=-1)
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

        if token_mask is not None:
            attn_mask = token_mask[:, None, None, :]
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(
            spike_q.unsqueeze(1),
            spike_k.unsqueeze(1),
            spike_v.unsqueeze(1),
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        ).squeeze(1)
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
    Hybrid tokenized spiking PPO policy with multi-timescale token memory.

    Pipeline:
      raw obs
        -> conv backbone
        -> pooled spatial tokens
        -> spiking self-attention
        -> fast + slow token temporal SNN pathways
        -> shared readout
        -> action/value heads

    Philosophy:
      The attention block resolves "what matters right now" within the
      current observation, while the temporal SNN pathways carry token-
      level memory across env steps. A faster pathway can react to local
      micro/combat changes, while a slower pathway can hold onto longer
      temporal context without forcing one leak timescale to do both jobs.
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
        fast_token_snn_alpha=None,
        fast_token_snn_beta=None,
        slow_token_snn_alpha=0.92,
        slow_token_snn_beta=0.97,
        temporal_combine_mode="mean",
        attention_embed_dim=64,
        attention_pool_size=7,
        attention_beta=0.5,
        spatial_head_type="token_pointer",
        coarse_grid_size=None,
        local_grid_size=None,
        target_decode_mode="center",
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
        self._carry_selection_state = False
        self._meta_input_dim = int(vector_input_dim)
        self._latent_dim = 64
        fast_token_snn_alpha = float(
            token_snn_alpha if fast_token_snn_alpha is None else fast_token_snn_alpha,
        )
        fast_token_snn_beta = float(
            token_snn_beta if fast_token_snn_beta is None else fast_token_snn_beta,
        )
        slow_token_snn_alpha = float(slow_token_snn_alpha)
        slow_token_snn_beta = float(slow_token_snn_beta)
        temporal_combine_mode = str(temporal_combine_mode).lower()
        if temporal_combine_mode not in {"mean", "sum"}:
            raise ValueError(
                "temporal_combine_mode must be 'mean' or 'sum', "
                f"got {temporal_combine_mode!r}",
            )
        self._temporal_pathways = 2
        self._temporal_combine_mode = temporal_combine_mode
        self._spatial_head_type = str(spatial_head_type).lower()
        self._coarse_grid_size = (
            self._pool_size if coarse_grid_size is None else int(coarse_grid_size)
        )
        self._local_grid_size = (
            self.screen_size // self._coarse_grid_size
            if local_grid_size is None
            else int(local_grid_size)
        )
        self._target_decode_mode = str(target_decode_mode).lower()
        if self._coarse_grid_size != self._pool_size:
            raise ValueError(
                "coarse_grid_size must match attention_pool_size for the current spatial tokenizer, "
                f"got coarse_grid_size={self._coarse_grid_size} and attention_pool_size={self._pool_size}",
            )
        self._config = {
            "num_steps": self.num_steps,
            "screen_size": self.screen_size,
            "fast_token_snn_alpha": fast_token_snn_alpha,
            "fast_token_snn_beta": fast_token_snn_beta,
            "slow_token_snn_alpha": slow_token_snn_alpha,
            "slow_token_snn_beta": slow_token_snn_beta,
            "temporal_pathways": int(self._temporal_pathways),
            "temporal_combine_mode": self._temporal_combine_mode,
            "attention_embed_dim": self._embed_dim,
            "attention_pool_size": self._pool_size,
            "attention_beta": float(attention_beta),
            "spatial_positional_encoding": "learned_xy_mlp",
            "spatial_head_type": self._spatial_head_type,
            "coarse_grid_size": int(self._coarse_grid_size),
            "local_grid_size": int(self._local_grid_size),
            "target_decode_mode": self._target_decode_mode,
            "meta_input_dim": self._meta_input_dim,
            "carry_entity_state": bool(self._carry_entity_state),
            "carry_selection_state": bool(self._carry_selection_state),
            "action_dim": int(action_dim),
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
        self.spatial_pos_proj = nn.Sequential(
            nn.Linear(2, self._embed_dim),
            nn.ReLU(),
            nn.Linear(self._embed_dim, self._embed_dim),
        )
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self._pool_size),
            torch.linspace(-1.0, 1.0, self._pool_size),
            indexing="ij",
        )
        spatial_pos_grid = torch.stack((grid_x, grid_y), dim=-1).view(
            1,
            self._spatial_tokens,
            2,
        )
        self.register_buffer(
            "spatial_pos_grid",
            spatial_pos_grid,
            persistent=False,
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
            alpha=fast_token_snn_alpha,
            beta=fast_token_snn_beta,
            spike_grad=spike_grad,
        )
        self.slow_token_snn = TokenTemporalSNN(
            alpha=slow_token_snn_alpha,
            beta=slow_token_snn_beta,
            spike_grad=spike_grad,
        )

        pooled_dim = TOKEN_TYPE_GROUPS * self._embed_dim
        self.combined_norm = nn.LayerNorm(pooled_dim)
        self.shared_fc1 = nn.Linear(pooled_dim, 128)
        self.shared_fc2 = nn.Linear(128, self._latent_dim)

        self._action_dim = int(action_dim)
        self.actor_fc = nn.Linear(self._latent_dim, action_dim)
        self.critic_fc = nn.Linear(self._latent_dim, 1)
        self.target_head = self._build_target_head_module()

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

    def _build_target_head_module(self) -> nn.Module:
        if self._spatial_head_type == "factorized_xy":
            return FactorizedXYTargetHead(
                embed_dim=self._embed_dim,
                latent_dim=self._latent_dim,
                action_dim=self._action_dim,
                screen_size=self.screen_size,
            )
        if self._spatial_head_type == "token_pointer":
            return TokenPointerTargetHead(
                embed_dim=self._embed_dim,
                latent_dim=self._latent_dim,
                action_dim=self._action_dim,
                coarse_grid_size=self._coarse_grid_size,
                local_grid_size=self._local_grid_size,
                screen_size=self.screen_size,
                target_decode_mode=self._target_decode_mode,
            )
        if self._spatial_head_type == "coarse_to_fine":
            return CoarseToFineTargetHead(
                embed_dim=self._embed_dim,
                latent_dim=self._latent_dim,
                action_dim=self._action_dim,
                coarse_grid_size=self._coarse_grid_size,
                local_grid_size=self._local_grid_size,
                screen_size=self.screen_size,
                target_decode_mode=self._target_decode_mode,
            )
        raise ValueError(
            f"Unsupported spatial_head_type: {self._spatial_head_type!r}",
        )

    def _zero_entity_state(
        self,
        syn_tok: torch.Tensor,
        mem_tok: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Zero recurrent state for token groups whose slot-to-identity
        # mapping is unstable across env steps (entity + selection).
        if self._carry_entity_state and self._carry_selection_state:
            return syn_tok, mem_tok
        syn_tok = syn_tok.clone()
        mem_tok = mem_tok.clone()
        if not self._carry_entity_state:
            if syn_tok.ndim == 3:
                syn_tok[:, self._entity_start : self._entity_end, :] = 0.0
                mem_tok[:, self._entity_start : self._entity_end, :] = 0.0
            else:
                syn_tok[:, :, self._entity_start : self._entity_end, :] = 0.0
                mem_tok[:, :, self._entity_start : self._entity_end, :] = 0.0
        if not self._carry_selection_state:
            if syn_tok.ndim == 3:
                syn_tok[:, self._selection_start : self._selection_end, :] = 0.0
                mem_tok[:, self._selection_start : self._selection_end, :] = 0.0
            else:
                syn_tok[:, :, self._selection_start : self._selection_end, :] = 0.0
                mem_tok[:, :, self._selection_start : self._selection_end, :] = 0.0
        return syn_tok, mem_tok

    def reset_state_rows(
        self,
        state: tuple[torch.Tensor, torch.Tensor] | None,
        reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if state is None or reset_mask is None:
            return state
        if reset_mask.ndim != 1:
            raise ValueError(
                f"reset_mask must be 1D over batch rows, got {tuple(reset_mask.shape)}",
            )
        if state[0].size(0) != int(reset_mask.numel()):
            raise ValueError(
                "reset_mask batch dimension must match recurrent state rows",
            )
        if not bool(reset_mask.any().item()):
            return state

        keep_mask = (~reset_mask).to(
            device=state[0].device,
            dtype=state[0].dtype,
        ).view((-1,) + (1,) * (state[0].ndim - 1))
        return state[0] * keep_mask, state[1] * keep_mask

    def init_concrete_state(self, batch_size=1, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        syn = torch.zeros(
            batch_size,
            self._temporal_pathways,
            self._num_tokens,
            self._embed_dim,
            device=device,
            dtype=dtype,
        )
        mem = torch.zeros_like(syn)
        return syn, mem

    def _coerce_temporal_state(
        self,
        state_in: tuple[torch.Tensor, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state_in is None:
            return self.init_concrete_state(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        syn_tok, mem_tok = state_in
        syn_tok = syn_tok.to(device=device, dtype=dtype)
        mem_tok = mem_tok.to(device=device, dtype=dtype)
        if syn_tok.ndim == 3 and mem_tok.ndim == 3:
            syn_tok = torch.stack((syn_tok, torch.zeros_like(syn_tok)), dim=1)
            mem_tok = torch.stack((mem_tok, torch.zeros_like(mem_tok)), dim=1)
        elif syn_tok.ndim != 4 or mem_tok.ndim != 4:
            raise ValueError(
                "Token temporal state must be rank-3 legacy or rank-4 multi-timescale, "
                f"got syn={tuple(syn_tok.shape)} mem={tuple(mem_tok.shape)}",
            )

        expected = (
            batch_size,
            self._temporal_pathways,
            self._num_tokens,
            self._embed_dim,
        )
        if tuple(syn_tok.shape) != expected or tuple(mem_tok.shape) != expected:
            raise ValueError(
                "Token temporal state shape mismatch, expected "
                f"{expected}, got syn={tuple(syn_tok.shape)} mem={tuple(mem_tok.shape)}",
            )
        return syn_tok, mem_tok

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

    def _group_masked_mean(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Masked-mean-pool per semantic group then concat. Permutation-
        # invariant within entity/selection groups whose slot-to-identity
        # mapping is unstable. Token-type embedding already injected
        # group identity upstream; concat preserves it here.
        group_slices = (
            (0, self._spatial_tokens),
            (self._entity_start, self._entity_end),
            (self._selection_start, self._selection_end),
            (self._meta_start, self._meta_start + 1),
        )
        summaries = []
        for start, end in group_slices:
            token_slice = tokens[:, start:end, :]
            mask_slice = mask[:, start:end].unsqueeze(-1).to(dtype=token_slice.dtype)
            summed = (token_slice * mask_slice).sum(dim=1)
            count = mask_slice.sum(dim=1).clamp_min(1.0)
            summaries.append(summed / count)
        return torch.cat(summaries, dim=-1)

    def _add_spatial_positional_encoding(
        self,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        coords = self.spatial_pos_grid.to(
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        ).expand(spatial_tokens.size(0), -1, -1)
        pos_emb = self.spatial_pos_proj(coords)
        return spatial_tokens + pos_emb

    def encode_step_tensors(
        self,
        spatial_obs: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        selection_features: torch.Tensor,
        selection_mask: torch.Tensor,
        meta_vec: torch.Tensor,
        state_in: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        spatial_input = spatial_obs
        syn_tok, mem_tok = self._coerce_temporal_state(
            state_in,
            batch_size=spatial_input.size(0),
            device=spatial_input.device,
            dtype=spatial_input.dtype,
        )
        syn_tok, mem_tok = self._zero_entity_state(syn_tok, mem_tok)

        x = F.relu(self.conv1(spatial_input))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        spatial_tokens = self.token_pool(x)
        spatial_tokens = spatial_tokens.flatten(2).transpose(1, 2)
        spatial_tokens = self._add_spatial_positional_encoding(spatial_tokens)
        batch_size = spatial_tokens.size(0)
        device = spatial_tokens.device
        spatial_mask = torch.ones(
            batch_size,
            self._spatial_tokens,
            dtype=torch.bool,
            device=device,
        )
        entity_tokens = self.entity_encoder(
            entity_features,
            entity_mask,
        )
        selection_tokens = self.selection_encoder(
            selection_features,
            selection_mask,
        )
        meta_tokens = self.meta_encoder(meta_vec)
        meta_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        tokens = torch.cat(
            (
                self._add_token_type(spatial_tokens, 0, spatial_mask),
                self._add_token_type(entity_tokens, 1, entity_mask),
                self._add_token_type(selection_tokens, 2, selection_mask),
                self._add_token_type(meta_tokens, 3, meta_mask),
            ),
            dim=1,
        )
        token_mask = torch.cat(
            (
                spatial_mask,
                entity_mask,
                selection_mask,
                meta_mask,
            ),
            dim=1,
        )
        attended = self.attention(tokens, token_mask=token_mask)

        spike_rec = []
        token_mask_f = token_mask.unsqueeze(-1).to(dtype=attended.dtype)
        pathway_token_mask_f = token_mask_f.unsqueeze(1)
        syn_tok = syn_tok * pathway_token_mask_f
        mem_tok = mem_tok * pathway_token_mask_f
        for _ in range(self.num_steps):
            fast_spk, fast_syn, fast_mem = self.token_snn(
                attended,
                syn_tok[:, 0],
                mem_tok[:, 0],
            )
            slow_spk, slow_syn, slow_mem = self.slow_token_snn(
                attended,
                syn_tok[:, 1],
                mem_tok[:, 1],
            )
            combined_spk = fast_spk + slow_spk
            if self._temporal_combine_mode == "mean":
                combined_spk = combined_spk * 0.5
            combined_spk = combined_spk * token_mask_f
            syn_tok = torch.stack((fast_syn, slow_syn), dim=1) * pathway_token_mask_f
            mem_tok = torch.stack((fast_mem, slow_mem), dim=1) * pathway_token_mask_f
            spike_rec.append(combined_spk)
        syn_tok, mem_tok = self._zero_entity_state(syn_tok, mem_tok)

        aggregated = torch.stack(spike_rec, dim=0).sum(dim=0)
        spatial_context = aggregated[:, : self._spatial_tokens, :]
        spatial_context = spatial_context.transpose(1, 2).reshape(
            batch_size,
            self._embed_dim,
            self._pool_size,
            self._pool_size,
        )
        pooled = self._group_masked_mean(aggregated, token_mask)
        combined = self.combined_norm(pooled)

        latent = F.relu(self.shared_fc1(combined))
        latent = F.relu(self.shared_fc2(latent))
        state_value = self.critic_fc(latent).squeeze(-1)
        next_state = (syn_tok, mem_tok)
        return latent, state_value, next_state, spatial_context

    def action_head(self, latent: torch.Tensor) -> torch.Tensor:
        return self.actor_fc(latent)

    def build_target_head(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor | None,
    ) -> TargetHeadState:
        if action_ids is None:
            action_ids = torch.full(
                (latent.size(0),),
                POLICY_ACTION_NO_OP,
                device=latent.device,
                dtype=torch.long,
            )
        else:
            action_ids = action_ids.to(device=latent.device, dtype=torch.long)
        action_ids = action_ids.clamp(0, self._action_dim - 1)
        return self.target_head.build(latent, spatial_context, action_ids)

    def sample_target(
        self,
        target_head_state: TargetHeadState,
        action_ids: torch.Tensor | None,
        deterministic: bool = False,
    ) -> TargetSample:
        del action_ids
        return self.target_head.sample(
            target_head_state,
            deterministic=deterministic,
        )

    def evaluate_target(
        self,
        target_head_state: TargetHeadState,
        recorded_target: dict[str, torch.Tensor | None],
        action_ids: torch.Tensor | None,
    ) -> TargetEval:
        del action_ids
        return self.target_head.evaluate(
            target_head_state,
            x=recorded_target["x"],
            y=recorded_target["y"],
            target_index=recorded_target.get("target_index"),
            coarse_index=recorded_target.get("coarse_index"),
            fine_index=recorded_target.get("fine_index"),
        )

    def decode_target_to_xy(
        self,
        *,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target_head.decode_target_to_xy(
            x=x,
            y=y,
            target_index=target_index,
            coarse_index=coarse_index,
            fine_index=fine_index,
        )

    def encode_xy_to_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        return self.target_head.encode_xy_to_target(x, y)

    def conditioned_spatial_head(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_head_state = self.build_target_head(latent, spatial_context, action_ids)
        if target_head_state.secondary_logits is None:
            raise RuntimeError(
                "conditioned_spatial_head compatibility is only available for factorized_xy",
            )
        return target_head_state.primary_logits, target_head_state.secondary_logits

    def forward_step_tensors(
        self,
        spatial_obs: torch.Tensor,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        selection_features: torch.Tensor,
        selection_mask: torch.Tensor,
        meta_vec: torch.Tensor,
        state_in: tuple[torch.Tensor, torch.Tensor] | None,
        action_ids: torch.Tensor | None = None,
    ):
        latent, state_value, next_state, spatial_context = self.encode_step_tensors(
            spatial_obs=spatial_obs,
            entity_features=entity_features,
            entity_mask=entity_mask,
            selection_features=selection_features,
            selection_mask=selection_mask,
            meta_vec=meta_vec,
            state_in=state_in,
        )
        action_logits = self.action_head(latent)
        if action_ids is None:
            action_ids = action_logits.float().argmax(dim=-1)
        target_head_state = self.build_target_head(
            latent,
            spatial_context,
            action_ids,
        )
        return action_logits, target_head_state, state_value, next_state

    def forward(self, batch: PolicyInputBatch, action_ids: torch.Tensor | None = None):
        if not isinstance(batch, PolicyInputBatch):
            raise TypeError(
                f"PolicyNetwork.forward expects PolicyInputBatch, got {type(batch)!r}",
            )
        return self.forward_step_tensors(
            spatial_obs=batch.spatial_obs,
            entity_features=batch.entity_features,
            entity_mask=batch.entity_mask,
            selection_features=batch.selection_features,
            selection_mask=batch.selection_mask,
            meta_vec=batch.meta_vec,
            state_in=batch.state_in,
            action_ids=action_ids,
        )
