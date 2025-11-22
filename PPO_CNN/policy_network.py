import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

class PolicyNetwork(nn.Module):
    """
    Spiking Neural Network adapted from the original CNN PolicyNetwork.
    
    Changes:
    - Replaces ReLU activations with Leaky Integrate-and-Fire (LIF) neurons.
    - Introduces a time-loop in the forward pass to simulate spiking dynamics.
    - Maintains exact input/output structure and layer dimensions of the original.
    """
    def __init__(self, spatial_input_shape, vector_input_dim, action_dim, num_steps=8, beta=0.9):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- SNN Specifics ---
        self.num_steps = num_steps  # Simulation steps per game step
        self.beta = beta            # Decay rate for LIF neurons
        spike_grad = surrogate.fast_sigmoid() # Gradient approximation for spikes

        # --- CNN Layers (Spiking) ---
        # Original: Conv2d -> Relu
        self.conv1 = nn.Conv2d(spatial_input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)

        # Original: Conv2d -> Relu -> Pool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)
        
        # Original: Conv2d -> Relu -> Pool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)
        
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size (Same logic as original)
        conv_out_size = (spatial_input_shape[1] // 2 // 2) * (spatial_input_shape[2] // 2 // 2) * 64

        # --- Shared FC Layers (Non-Spiking / Readout) ---
        # These remain standard ANN layers to ensure stable Actor/Critic values
        self.shared_fc1 = nn.Linear(conv_out_size + vector_input_dim, 128)
        self.shared_fc2 = nn.Linear(128, 64)

        # --- Output Heads (Identical to Original) ---
        self.actor_fc = nn.Linear(64, action_dim)
        self.critic_fc = nn.Linear(64, 1)
        self.angle_fc = nn.Linear(64, 4)

        # AMP configuration
        self.amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))

        self.to(self.device)

    def forward(self, spatial_input, vector_input, state=None):
        batch_size = spatial_input.size(0)

        # 1. Handle State Initialization
        # If this is the first frame (state is None), initialize fresh membranes
        if state is None:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
        else:
            # Unpack the saved state from the previous frame
            mem1, mem2, mem3 = state

        spk3_rec = [] 

        # --- SNN Time Loop ---
        # We still run the internal loop, but now we start with PREVIOUS memory
        for step in range(self.num_steps):
            cur1 = self.conv1(spatial_input) 
            spk1, mem1 = self.lif1(cur1, mem1) # Updates mem1
            
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2) # Updates mem2
            x2 = self.pool(spk2)
            
            cur3 = self.conv3(x2)
            spk3, mem3 = self.lif3(cur3, mem3) # Updates mem3
            x3 = self.pool(spk3)
            
            spk3_rec.append(x3)

        # --- Aggregation ---
        # Stack outputs: [Steps, Batch, Channels, H, W]
        spk3_stacked = torch.stack(spk3_rec, dim=0)
        
        # Sum spikes over time (Rate Coding) to get a dense feature map
        cnn_features = spk3_stacked.sum(dim=0) 
        
        # Flatten (Matches original: torch.flatten(x, start_dim=1))
        x = torch.flatten(cnn_features, start_dim=1)

        # --- Feature Combination & Output Heads (Identical to Original) ---
        combined = torch.cat([x, vector_input], dim=1)
        
        x = F.relu(self.shared_fc1(combined))
        x = F.relu(self.shared_fc2(x))

        action_logits = self.actor_fc(x)
        angle_logits = self.angle_fc(x)
        state_value = self.critic_fc(x).squeeze(-1)

        # Identical angle calculation
        angle = torch.atan2(angle_logits[:, 1], angle_logits[:, 0])
        next_state = (mem1.detach(), mem2.detach(), mem3.detach())
        return action_logits, angle, state_value, next_state