import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


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

    def forward(self, tokens):
        tokens_normed = self.pre_norm(tokens)

        q_raw = self.W_q(tokens_normed)
        k_raw = self.W_k(tokens_normed)
        v_raw = self.W_v(tokens_normed)

        mem_q = self.lif_q.init_leaky()
        mem_k = self.lif_k.init_leaky()
        mem_v = self.lif_v.init_leaky()

        spike_q, _ = self.lif_q(q_raw, mem_q)
        spike_k, _ = self.lif_k(k_raw, mem_k)
        spike_v, _ = self.lif_v(v_raw, mem_v)

        attn = torch.bmm(spike_q, spike_k.transpose(1, 2)) * self.scale
        out = torch.bmm(attn, spike_v)
        return out + tokens


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
        self._num_tokens = self._pool_size * self._pool_size
        self._config = {
            "num_steps": self.num_steps,
            "screen_size": self.screen_size,
            "token_snn_alpha": float(token_snn_alpha),
            "token_snn_beta": float(token_snn_beta),
            "attention_embed_dim": self._embed_dim,
            "attention_pool_size": self._pool_size,
            "attention_beta": float(attention_beta),
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

        fc_input_dim = self._num_tokens * self._embed_dim + vector_input_dim
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
            "amp_enabled": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype),
            "device": str(self.device),
        }

    def init_concrete_state(self, batch_size=1, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return TokenTemporalSNN.init_state(
            batch_size, self._num_tokens, self._embed_dim, device, dtype,
        )

    def forward(self, spatial_input, vector_input, state=None):
        if state is None:
            syn_tok, mem_tok = self.init_concrete_state(
                batch_size=spatial_input.size(0),
                device=spatial_input.device,
                dtype=spatial_input.dtype,
            )
        else:
            syn_tok, mem_tok = state

        x = F.relu(self.conv1(spatial_input))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        tokens = self.token_pool(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        attended = self.attention(tokens)

        spike_rec = []
        for _ in range(self.num_steps):
            spk_tok, syn_tok, mem_tok = self.token_snn(attended, syn_tok, mem_tok)
            spike_rec.append(spk_tok)

        aggregated = torch.stack(spike_rec, dim=0).sum(dim=0)
        x = aggregated.flatten(start_dim=1)
        combined = torch.cat([x, vector_input], dim=1)
        combined = self.combined_norm(combined)

        x = F.relu(self.shared_fc1(combined))
        x = F.relu(self.shared_fc2(x))

        action_logits = self.actor_fc(x)
        move_x_logits = self.move_x_fc(x)
        move_y_logits = self.move_y_fc(x)
        state_value = self.critic_fc(x).squeeze(-1)
        next_state = (syn_tok, mem_tok)
        return action_logits, move_x_logits, move_y_logits, state_value, next_state
