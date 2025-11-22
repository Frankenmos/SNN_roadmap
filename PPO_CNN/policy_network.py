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
    def __init__(self, spatial_input_shape, vector_input_dim, action_dim, num_steps=8, beta=0.9):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_steps = num_steps
        # Make beta a learnable parameter per layer? 
        # Let's start simple: Fixed, but you can switch to learnable later.
        self.beta = beta 
        spike_grad = surrogate.fast_sigmoid(slope=25) # Sharper slope for better gradients

        # --- SNN Encoder ---
        self.conv1 = nn.Conv2d(spatial_input_shape[0], 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)
        
        self.pool = nn.MaxPool2d(2, 2)

        # --- Attention Mechanism ---
        # Applied to the final feature map before flattening
        self.attention = SpatialAttention()

        # --- Flatten Calculation ---
        # Calculate output size based on pooling structure
        # (Assuming input is roughly 32x32 or 64x64, div by 4 due to 2 pools)
        h_out = spatial_input_shape[1] // 4
        w_out = spatial_input_shape[2] // 4
        conv_out_size = h_out * w_out * 64

        # --- Integration Layer (LayerNorm is CRITICAL for PPO) ---
        self.ln = nn.LayerNorm(conv_out_size)

        # --- ANN Decoder ---
        self.shared_fc1 = nn.Linear(conv_out_size + vector_input_dim, 256) # Increased width
        self.shared_fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.1) # Light regularization

        # Heads
        self.actor_fc = nn.Linear(128, action_dim)
        self.critic_fc = nn.Linear(128, 1)
        self.angle_fc = nn.Linear(128, 2) # Changed to 2 (Sin/Cos or Vector)

        self.to(self.device)

    def forward(self, spatial_input, vector_input, state=None):
        batch_size = spatial_input.size(0)

        # Initialize or unpack state
        if state is None:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
        else:
            mem1, mem2, mem3 = state

        spk3_rec = []

        # --- 1. Input Normalization for Spiking ---
        # Ensure input is 0-1 for probability use
        # If spatial_input is standard PySC2 (0-255 or raw), normalize it here.
        # Assuming spatial_input is float32.
        # We clamp to ensure stable probabilities.
        x_prob = torch.clamp(spatial_input, 0.0, 1.0)

        # --- 2. SNN Loop ---
        for step in range(self.num_steps):
            # IMPROVEMENT: Bernoulli Encoding
            # We generate a new spike mask every time step based on input intensity
            # This creates "noise" that represents the signal
            spike_input = torch.bernoulli(x_prob)
            
            # Layer 1
            cur1 = self.conv1(spike_input)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2 (Pool -> Conv)
            x1 = self.pool(spk1)
            cur2 = self.conv2(x1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Layer 3 (Pool -> Conv)
            x2 = self.pool(spk2)
            cur3 = self.conv3(x2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)

        # --- 3. Aggregation (Mean instead of Sum) ---
        # [Steps, Batch, C, H, W]
        spk3_stacked = torch.stack(spk3_rec, dim=0)
        # Mean creates a "Firing Rate Map" (0.0 to 1.0)
        cnn_features = spk3_stacked.mean(dim=0) 

        # --- 4. Spatial Attention ---
        # Highlights important regions in the firing rate map
        cnn_features = self.attention(cnn_features)

        # Flatten and Normalize
        x = torch.flatten(cnn_features, start_dim=1)
        x = self.ln(x) # LayerNorm stabilizes the transition from Spikes to ANN

        # --- 5. ANN Decode ---
        combined = torch.cat([x, vector_input], dim=1)
        
        x = F.relu(self.shared_fc1(combined))
        x = self.dropout(x) # Prevent overfitting to specific map features
        x = F.relu(self.shared_fc2(x))

        # Heads
        action_logits = self.actor_fc(x)
        angle_vec = self.angle_fc(x) 
        state_value = self.critic_fc(x).squeeze(-1)

        # Simplified Angle: Predict a vector and get atan2
        angle = torch.atan2(angle_vec[:, 1], angle_vec[:, 0])

        next_state = (mem1.detach(), mem2.detach(), mem3.detach())
        
        return action_logits, angle, state_value, next_state