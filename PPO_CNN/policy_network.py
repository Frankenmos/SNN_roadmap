import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class SpatialAttention(nn.Module):
    """
    Focuses on 'where' the informative features are.
    Useful for PySC2 to highlight units vs empty terrain.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Compresses channels to 1, creating a 2D attention map
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Channels, H, W]
        # MaxPool across channels (most prominent feature)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # AvgPool across channels (average feature intensity)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Concat to get spatial statistics
        y = torch.cat([max_pool, avg_pool], dim=1)
        # Generate attention map
        attn_map = self.sigmoid(self.conv(y))
        # Scale original features
        return x * attn_map

class PolicyNetwork(nn.Module):
    def __init__(self, spatial_input_shape, vector_input_dim, action_dim, num_steps=16):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = num_steps

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Different time constants → deeper layers remember longer
        self.conv1 = nn.Conv2d(spatial_input_shape[0], 16, 3, padding=1)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.lif2 = snn.Leaky(beta=0.90, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.lif3 = snn.Leaky(beta=0.85, spike_grad=spike_grad)

        self.pool = nn.MaxPool2d(2, 2)
        self.attention = SpatialAttention()

        # Feature size after two pools
        h_out = spatial_input_shape[1] // 4
        w_out = spatial_input_shape[2] // 4
        conv_out_size = 64 * h_out * w_out

        self.ln = nn.LayerNorm(conv_out_size)

        # Shared trunk
        self.fc1 = nn.Linear(conv_out_size + vector_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.1)

        # Heads
        self.action_head = nn.Linear(128, action_dim)      # Discrete actions
        self.xy_head = nn.Linear(128, 2)                   # Predicted click (x,y) in [0,1]
        self.value_head = nn.Linear(128, 1)

        self.to(self.device)

    def forward(self, spatial_input, vector_input, state=None):
        batch_size = spatial_input.size(0)
        x_prob = torch.clamp(spatial_input, 0.0, 1.0)

        if state is None:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
        else:
            mem1, mem2, mem3 = state

        spk_rec = []

        for _ in range(self.num_steps):
            spk_in = torch.bernoulli(x_prob)

            cur1 = self.conv1(spk_in)
            spk1, mem1 = self.lif1(cur1, mem1)

            x = self.pool(spk1)
            cur2 = self.conv2(x)
            spk2, mem2 = self.lif2(cur2, mem2)

            x = self.pool(spk2)
            cur3 = self.conv3(x)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_rec.append(spk3)

        rate_map = torch.stack(spk_rec, dim=0).sum(0) / self.num_steps   # [B,C,H,W]
        rate_map = self.attention(rate_map)
        x = torch.flatten(rate_map, start_dim=1)
        x = self.ln(x)

        x = torch.cat([x, vector_input], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)          # Discrete actions
        xy_raw = self.xy_head(x)                     # raw output
        xy = torch.sigmoid(xy_raw)                   # force to [0,1] screen coordinates
        value = self.value_head(x).squeeze(-1)

        next_state = (mem1.detach(), mem2.detach(), mem3.detach())
        return action_logits, xy, value, next_state