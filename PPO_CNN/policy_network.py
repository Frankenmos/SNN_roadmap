
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class SpikingSelfAttention(nn.Module):
    """
    Spikformer-style spike-driven self-attention.

    Instead of softmax(QK^T/sqrt(d))V, computes:
        spike(Q) @ spike(K)^T * scale @ spike(V)

    Q, K, V are encoded as binary spike trains via LIF neurons,
    making the attention computation multiplication-free in the
    inner product (just additions of 0s and 1s).
    """

    def __init__(self, embed_dim=64, beta_qkv=0.5, spike_grad=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        # Linear projections (no bias — Spikformer convention)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # LIF neurons for spike-encoding Q, K, V
        # Low beta = fast decay = responsive binary encoding
        self.lif_q = snn.Leaky(beta=beta_qkv, spike_grad=spike_grad, learn_beta=True)
        self.lif_k = snn.Leaky(beta=beta_qkv, spike_grad=spike_grad, learn_beta=True)
        self.lif_v = snn.Leaky(beta=beta_qkv, spike_grad=spike_grad, learn_beta=True)

    def init_state(self):
        """Returns (mem_q, mem_k, mem_v) initialized to zero tensors."""
        return (
            self.lif_q.init_leaky(),
            self.lif_k.init_leaky(),
            self.lif_v.init_leaky(),
        )

    def forward(self, tokens, mem_q, mem_k, mem_v):
        """
        Args:
            tokens: [B, N, D] input token features
            mem_q, mem_k, mem_v: LIF membrane potentials from previous timestep

        Returns:
            out: [B, N, D] attended + residual features
            mem_q, mem_k, mem_v: updated membrane potentials
        """
        # Project to Q, K, V
        q_raw = self.W_q(tokens)  # [B, N, D]
        k_raw = self.W_k(tokens)  # [B, N, D]
        v_raw = self.W_v(tokens)  # [B, N, D]

        # Spike-encode via LIF neurons
        spike_q, mem_q = self.lif_q(q_raw, mem_q)  # binary [B, N, D]
        spike_k, mem_k = self.lif_k(k_raw, mem_k)  # binary [B, N, D]
        spike_v, mem_v = self.lif_v(v_raw, mem_v)  # binary [B, N, D]

        # Spike-driven attention (no softmax!)
        # spike_q @ spike_k^T: counts how many dimensions both fire
        attn = torch.bmm(spike_q, spike_k.transpose(1, 2)) * self.scale  # [B, N, N]

        # Attend to spike-encoded values
        out = torch.bmm(attn, spike_v)  # [B, N, D]

        # Residual connection for gradient flow
        out = out + tokens

        return out, mem_q, mem_k, mem_v


class PolicyNetwork(nn.Module):
    """
    Spiking Transformer PolicyNetwork with multi-temporal dynamics.

    Evolution from the original SNN PolicyNetwork:
    - Multi-temporal conv backbone: Synaptic (fast), Synaptic (medium), Leaky (slow)
    - Learnable time constants (alpha, beta) per layer
    - Spike-driven self-attention (Spikformer-style) over spatial tokens
    - Adaptive pooling for spatial tokenization
    - Standard ANN readout heads for stable RL training
    """

    def __init__(self, spatial_input_shape, vector_input_dim, action_dim, num_steps=8):
        super().__init__()

        # Choose device safely
        try:
            if torch.cuda.is_available():
                try:
                    _ = torch.tensor([0.0], device=torch.device("cuda"))
                    device = torch.device("cuda")
                except Exception:
                    warnings.warn("CUDA appears available but failed a test allocation. Falling back to CPU.")
                    device = torch.device("cpu")
            else:
                device = torch.device("cpu")
        except Exception:
            device = torch.device("cpu")

        self.device = device
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()

        # === Multi-Temporal Conv Backbone ===

        # Layer 1: Fast interneurons — rapid synaptic decay, medium membrane
        # Low alpha = synaptic current decays quickly (transient edge detection)
        self.conv1 = nn.Conv2d(spatial_input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.snn1 = snn.Synaptic(
            alpha=0.5, beta=0.8, spike_grad=spike_grad,
            learn_alpha=True, learn_beta=True,
        )

        # Layer 2: Medium pyramidal cells — slower synaptic filtering
        # Higher alpha = synaptic current persists (temporal bandpass)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.snn2 = snn.Synaptic(
            alpha=0.8, beta=0.9, spike_grad=spike_grad,
            learn_alpha=True, learn_beta=True,
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Layer 3: Slow integrator — persistent working memory
        # High beta = membrane barely decays (evidence accumulation)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.snn3 = snn.Leaky(
            beta=0.95, spike_grad=spike_grad, learn_beta=True,
        )

        # === Spatial Tokenization ===
        # Pool conv output to fixed grid, then treat as sequence of tokens
        self.token_pool = nn.AdaptiveAvgPool2d((7, 7))
        num_tokens = 7 * 7  # 49 spatial tokens
        embed_dim = 64       # channels become token embedding dim

        # === Spiking Self-Attention ===
        self.attention = SpikingSelfAttention(
            embed_dim=embed_dim, beta_qkv=0.5, spike_grad=spike_grad,
        )

        # === Shared FC Layers (Non-Spiking / Readout) ===
        fc_input_dim = num_tokens * embed_dim + vector_input_dim  # 3136 + 100 = 3236
        self.shared_fc1 = nn.Linear(fc_input_dim, 128)
        self.shared_fc2 = nn.Linear(128, 64)

        # === Output Heads ===
        self.actor_fc = nn.Linear(64, action_dim)
        self.critic_fc = nn.Linear(64, 1)
        self.angle_fc = nn.Linear(64, 4)

        # AMP configuration
        self.amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))

        self.to(self.device)

    def forward(self, spatial_input, vector_input, state=None):
        """
        Args:
            spatial_input: [B, 27, 84, 84] feature screen
            vector_input:  [B, 100] vector observations
            state: tuple of neuron states or None (fresh init)

        Returns:
            action_logits: [B, action_dim]
            angle: [B]
            state_value: [B]
            next_state: detached neuron states for next timestep
        """

        # === 1. State Initialization ===
        if state is None:
            syn1, mem1 = self.snn1.init_synaptic()
            syn2, mem2 = self.snn2.init_synaptic()
            mem3 = self.snn3.init_leaky()
            mem_q, mem_k, mem_v = self.attention.init_state()
        else:
            (syn1, mem1), (syn2, mem2), mem3, (mem_q, mem_k, mem_v) = state

        # === 2. SNN Time Loop ===
        attn_out_rec = []

        for step in range(self.num_steps):
            # Layer 1: fast synaptic dynamics
            cur1 = self.conv1(spatial_input)
            spk1, syn1, mem1 = self.snn1(cur1, syn1, mem1)

            # Layer 2: medium synaptic dynamics + pooling
            cur2 = self.conv2(spk1)
            spk2, syn2, mem2 = self.snn2(cur2, syn2, mem2)
            x2 = self.pool(spk2)

            # Layer 3: slow leaky integration + pooling
            cur3 = self.conv3(x2)
            spk3, mem3 = self.snn3(cur3, mem3)
            x3 = self.pool(spk3)  # [B, 64, 21, 21]

            # Spatial tokenization
            tokens = self.token_pool(x3)                     # [B, 64, 7, 7]
            tokens = tokens.flatten(2).transpose(1, 2)       # [B, 49, 64]

            # Spiking self-attention
            attn_out, mem_q, mem_k, mem_v = self.attention(tokens, mem_q, mem_k, mem_v)
            attn_out_rec.append(attn_out)

        # === 3. Temporal Aggregation (rate coding) ===
        stacked = torch.stack(attn_out_rec, dim=0)  # [T, B, 49, 64]
        aggregated = stacked.sum(dim=0)               # [B, 49, 64]

        # Flatten tokens
        x = aggregated.flatten(start_dim=1)  # [B, 3136]

        # === 4. Feature Combination & Output Heads ===
        combined = torch.cat([x, vector_input], dim=1)  # [B, 3236]

        x = F.relu(self.shared_fc1(combined))
        x = F.relu(self.shared_fc2(x))

        action_logits = self.actor_fc(x)
        angle_logits = self.angle_fc(x)
        state_value = self.critic_fc(x).squeeze(-1)

        angle = torch.atan2(angle_logits[:, 1], angle_logits[:, 0])

        # === 5. State Packaging (all detached) ===
        next_state = (
            (syn1.detach(), mem1.detach()),
            (syn2.detach(), mem2.detach()),
            mem3.detach(),
            (mem_q.detach(), mem_k.detach(), mem_v.detach()),
        )

        return action_logits, angle, state_value, next_state
