
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    A policy network for the PPO agent that processes both spatial (CNN) and
    vector inputs. The network has a shared backbone and separate output heads
    for the actor (action selection), critic (state value estimation), and a
    specialized head for predicting a continuous angle for movement.

    Architecture:
    - CNN Layers: Three convolutional layers with max pooling to extract features
      from the spatial input (e.g., game screen).
    - Shared FC Layers: The flattened output from the CNN is concatenated with
      the vector input and passed through two fully connected layers.
    - Output Heads:
        - Actor Head: A linear layer that outputs logits for the discrete action space.
        - Critic Head: A linear layer that outputs a single value representing the state value.
        - Angle Head: A linear layer that outputs four values, which are used to
          compute a continuous angle using a circular mapping (atan2).
    """
    def __init__(self, spatial_input_shape, vector_input_dim, action_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- CNN Layers for Spatial Feature Extraction ---
        # These layers process the game's screen data to identify spatial patterns.
        self.conv1 = nn.Conv2d(spatial_input_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the flattened size of the CNN output after pooling.
        conv_out_size = (spatial_input_shape[1] // 2 // 2) * (spatial_input_shape[2] // 2 // 2) * 64

        # --- Shared Fully Connected Layers ---
        # These layers form a shared backbone for both the actor and the critic.
        # They process the combined information from both spatial and vector inputs.
        self.shared_fc1 = nn.Linear(conv_out_size + vector_input_dim, 128)
        self.shared_fc2 = nn.Linear(128, 64)

        # --- Output Heads ---
        # Actor Head: Outputs logits for the discrete action distribution.
        self.actor_fc = nn.Linear(64, action_dim)
        # Critic Head: Outputs a single value representing the estimated state value.
        self.critic_fc = nn.Linear(64, 1)
        # Angle Head: Outputs values for calculating a continuous angle for movement.
        self.angle_fc = nn.Linear(64, 4)

        # AMP (Automatic Mixed Precision) configuration for performance.
        self.amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))

        self.to(self.device)

    def forward(self, spatial_input, vector_input):
        """
        Performs the forward pass through the network.

        Args:
            spatial_input (torch.Tensor): The spatial observation from the environment.
            vector_input (torch.Tensor): The vector observation from the environment.

        Returns:
            tuple: A tuple containing:
                - action_logits (torch.Tensor): Logits for the action distribution.
                - angle (torch.Tensor): The calculated continuous angle.
                - state_value (torch.Tensor): The estimated value of the state.
        """
        # --- CNN Processing ---
        # Pass the spatial input through the convolutional layers.
        x = F.relu(self.conv1(spatial_input))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the output for the fully connected layers.
        x = torch.flatten(x, start_dim=1)

        # --- Feature Combination ---
        # Combine the processed spatial features with the vector input.
        combined = torch.cat([x, vector_input], dim=1)
        x = F.relu(self.shared_fc1(combined))
        x = F.relu(self.shared_fc2(x))

        # --- Output Head Forward Passes ---
        action_logits = self.actor_fc(x)
        angle_logits = self.angle_fc(x)  # Shape: [B, 4]
        state_value = self.critic_fc(x).squeeze(-1)  # Shape: [B]

        # --- Angle Calculation ---
        # Compute a continuous angle using a circular mapping (atan2).
        # This is a common technique to represent angles in a way that is
        # friendly to neural networks, as it handles the wrap-around nature
        # of angles (e.g., 360 degrees is the same as 0 degrees).
        angle = torch.atan2(angle_logits[:, 1], angle_logits[:, 0])

        return action_logits, angle, state_value
