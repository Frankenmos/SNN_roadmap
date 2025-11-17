
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, spatial_input_shape, vector_input_dim, action_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ...

        # CNN layers
        self.conv1 = nn.Conv2d(spatial_input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Flattened size
        conv_out_size = (spatial_input_shape[1] // 2 // 2) * (spatial_input_shape[2] // 2 // 2) * 64

        # Shared layers
        self.shared_fc1 = nn.Linear(conv_out_size + vector_input_dim, 128)
        self.shared_fc2 = nn.Linear(128, 64)

        # Output heads
        self.actor_fc = nn.Linear(64, action_dim)
        self.critic_fc = nn.Linear(64, 1)
        self.angle_fc = nn.Linear(64, 4)

        # AMP config
        self.amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        self.to(self.device)

    def forward(self, spatial_input, vector_input):
        # CNN processing
        x = F.relu(self.conv1(spatial_input))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)

        # Combine spatial and vector inputs
        combined = torch.cat([x, vector_input], dim=1)
        x = F.relu(self.shared_fc1(combined))
        x = F.relu(self.shared_fc2(x))

        # Output heads
        action_logits = self.actor_fc(x)
        angle_logits = self.angle_fc(x)  # Shape: [B, 4]
        state_value = self.critic_fc(x).squeeze(-1)  # Shape: [B]

        # Compute angle using circular mapping
        angle = torch.atan2(angle_logits[:, 1], angle_logits[:, 0])  # Use 2 components for sine/cosine

        return action_logits, angle, state_value
