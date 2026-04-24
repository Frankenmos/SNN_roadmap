from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class TargetHeadState:
    head_type: str
    primary_logits: torch.Tensor
    secondary_logits: torch.Tensor | None = None


@dataclass(slots=True)
class TargetSample:
    x: torch.Tensor
    y: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    target_index: torch.Tensor | None = None
    coarse_index: torch.Tensor | None = None
    fine_index: torch.Tensor | None = None


@dataclass(slots=True)
class TargetEval:
    log_prob: torch.Tensor
    entropy: torch.Tensor


class BaseSpatialTargetHead(nn.Module):
    head_type = "base"

    def build(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> TargetHeadState:
        raise NotImplementedError

    def sample(
        self,
        head_state: TargetHeadState,
        deterministic: bool = False,
    ) -> TargetSample:
        raise NotImplementedError

    def evaluate(
        self,
        head_state: TargetHeadState,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> TargetEval:
        raise NotImplementedError

    def encode_xy_to_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        raise NotImplementedError

    def decode_target_to_xy(
        self,
        *,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _normalized_entropy(entropy: torch.Tensor, num_classes: int) -> torch.Tensor:
        if int(num_classes) <= 1:
            return torch.zeros_like(entropy)
        return entropy / math.log(float(num_classes))


class FactorizedXYTargetHead(BaseSpatialTargetHead):
    head_type = "factorized_xy"

    def __init__(
        self,
        *,
        embed_dim: int,
        latent_dim: int,
        action_dim: int,
        screen_size: int,
    ) -> None:
        super().__init__()
        self.screen_size = int(screen_size)
        self.embed_dim = int(embed_dim)
        self.action_condition_embedding = nn.Embedding(int(action_dim), latent_dim)
        self.latent_to_spatial = nn.Linear(int(latent_dim), self.embed_dim)
        self.action_to_spatial = nn.Linear(int(latent_dim), self.embed_dim)
        self.click_tower = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.x_readout = nn.Linear(self.embed_dim, 1)
        self.y_readout = nn.Linear(self.embed_dim, 1)

    def build(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> TargetHeadState:
        action_ids = action_ids.clamp(
            0,
            self.action_condition_embedding.num_embeddings - 1,
        )
        action_emb = self.action_condition_embedding(action_ids)
        action_bias = self.action_to_spatial(action_emb).view(
            latent.size(0),
            self.embed_dim,
            1,
            1,
        )
        latent_bias = self.latent_to_spatial(latent).view(
            latent.size(0),
            self.embed_dim,
            1,
            1,
        )
        click_features = self.click_tower(spatial_context + action_bias + latent_bias)
        click_features = F.interpolate(
            click_features,
            size=(self.screen_size, self.screen_size),
            mode="bilinear",
            align_corners=False,
        )

        x_features = click_features.mean(dim=2).transpose(1, 2)
        y_features = click_features.mean(dim=3).transpose(1, 2)
        return TargetHeadState(
            head_type=self.head_type,
            primary_logits=self.x_readout(x_features).squeeze(-1),
            secondary_logits=self.y_readout(y_features).squeeze(-1),
        )

    def sample(
        self,
        head_state: TargetHeadState,
        deterministic: bool = False,
    ) -> TargetSample:
        x_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        y_dist = torch.distributions.Categorical(logits=head_state.secondary_logits.float())
        if deterministic:
            x = head_state.primary_logits.float().argmax(dim=-1)
            y = head_state.secondary_logits.float().argmax(dim=-1)
        else:
            x = x_dist.sample()
            y = y_dist.sample()
        entropy = self._normalized_entropy(
            x_dist.entropy(),
            head_state.primary_logits.size(-1),
        ) + self._normalized_entropy(
            y_dist.entropy(),
            head_state.secondary_logits.size(-1),
        )
        return TargetSample(
            x=x,
            y=y,
            log_prob=x_dist.log_prob(x) + y_dist.log_prob(y),
            entropy=entropy,
        )

    def evaluate(
        self,
        head_state: TargetHeadState,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> TargetEval:
        del target_index, coarse_index, fine_index
        x_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        y_dist = torch.distributions.Categorical(logits=head_state.secondary_logits.float())
        entropy = self._normalized_entropy(
            x_dist.entropy(),
            head_state.primary_logits.size(-1),
        ) + self._normalized_entropy(
            y_dist.entropy(),
            head_state.secondary_logits.size(-1),
        )
        return TargetEval(
            log_prob=x_dist.log_prob(x.long()) + y_dist.log_prob(y.long()),
            entropy=entropy,
        )

    def encode_xy_to_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        return {
            "x": x.long(),
            "y": y.long(),
            "target_index": None,
            "coarse_index": None,
            "fine_index": None,
        }

    def decode_target_to_xy(
        self,
        *,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_index, coarse_index, fine_index
        if x is None or y is None:
            raise ValueError("factorized_xy decode requires x and y tensors")
        return x.long(), y.long()


class TokenPointerTargetHead(BaseSpatialTargetHead):
    head_type = "token_pointer"

    def __init__(
        self,
        *,
        embed_dim: int,
        latent_dim: int,
        action_dim: int,
        coarse_grid_size: int,
        local_grid_size: int,
        screen_size: int,
        target_decode_mode: str = "center",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.coarse_grid_size = int(coarse_grid_size)
        self.local_grid_size = int(local_grid_size)
        self.screen_size = int(screen_size)
        self.target_decode_mode = str(target_decode_mode).lower()
        if self.target_decode_mode != "center":
            raise ValueError(
                f"Unsupported target_decode_mode: {target_decode_mode!r}",
            )
        if self.coarse_grid_size <= 0:
            raise ValueError("coarse_grid_size must be positive")
        if self.local_grid_size <= 0:
            raise ValueError("local_grid_size must be positive")
        if self.coarse_grid_size * self.local_grid_size != self.screen_size:
            raise ValueError(
                "Token-pointer head expects coarse_grid_size * local_grid_size == screen_size, "
                f"got {self.coarse_grid_size} * {self.local_grid_size} != {self.screen_size}",
            )
        self.action_condition_embedding = nn.Embedding(int(action_dim), self.latent_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    @property
    def token_count(self) -> int:
        return self.coarse_grid_size * self.coarse_grid_size

    def build(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> TargetHeadState:
        batch_size, embed_dim, height, width = spatial_context.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Token-pointer expected embed_dim={self.embed_dim}, got {embed_dim}",
            )
        if height != self.coarse_grid_size or width != self.coarse_grid_size:
            raise ValueError(
                "Token-pointer grid mismatch: "
                f"expected {self.coarse_grid_size}x{self.coarse_grid_size}, got {height}x{width}",
            )
        action_ids = action_ids.clamp(
            0,
            self.action_condition_embedding.num_embeddings - 1,
        )
        action_emb = self.action_condition_embedding(action_ids)
        query = self.query_mlp(torch.cat((latent, action_emb), dim=-1))
        tokens = spatial_context.flatten(2).transpose(1, 2)
        scores = torch.einsum("bd,bnd->bn", query, self.token_proj(tokens))
        if int(scores.size(0)) != int(batch_size):
            raise RuntimeError("Token-pointer build produced an invalid batch size")
        return TargetHeadState(head_type=self.head_type, primary_logits=scores)

    def sample(
        self,
        head_state: TargetHeadState,
        deterministic: bool = False,
    ) -> TargetSample:
        target_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        if deterministic:
            target_index = head_state.primary_logits.float().argmax(dim=-1)
        else:
            target_index = target_dist.sample()
        x, y = self.decode_target_to_xy(target_index=target_index)
        return TargetSample(
            x=x,
            y=y,
            target_index=target_index,
            coarse_index=None,
            fine_index=None,
            log_prob=target_dist.log_prob(target_index),
            entropy=self._normalized_entropy(
                target_dist.entropy(),
                head_state.primary_logits.size(-1),
            ),
        )

    def evaluate(
        self,
        head_state: TargetHeadState,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> TargetEval:
        del coarse_index, fine_index
        if target_index is None:
            encoded = self.encode_xy_to_target(x.long(), y.long())
            target_index = encoded["target_index"]
        else:
            target_index = target_index.long()
            if bool((target_index < 0).any().item()):
                encoded = self.encode_xy_to_target(x.long(), y.long())
                fallback_index = encoded["target_index"].long()
                target_index = torch.where(
                    target_index >= 0,
                    target_index,
                    fallback_index,
                )
        target_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        return TargetEval(
            log_prob=target_dist.log_prob(target_index.long()),
            entropy=self._normalized_entropy(
                target_dist.entropy(),
                head_state.primary_logits.size(-1),
            ),
        )

    def encode_xy_to_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        x = x.long().clamp(0, self.screen_size - 1)
        y = y.long().clamp(0, self.screen_size - 1)
        coarse_col = torch.div(x, self.local_grid_size, rounding_mode="floor")
        coarse_row = torch.div(y, self.local_grid_size, rounding_mode="floor")
        target_index = coarse_row * self.coarse_grid_size + coarse_col
        return {
            "x": x,
            "y": y,
            "target_index": target_index.long(),
            "coarse_index": None,
            "fine_index": None,
        }

    def decode_target_to_xy(
        self,
        *,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del x, y, coarse_index, fine_index
        if target_index is None:
            raise ValueError("token_pointer decode requires target_index")
        target_index = target_index.long().clamp(0, self.token_count - 1)
        row = torch.div(target_index, self.coarse_grid_size, rounding_mode="floor")
        col = torch.remainder(target_index, self.coarse_grid_size)
        center_offset = self.local_grid_size // 2
        x_out = (col * self.local_grid_size + center_offset).clamp(
            0,
            self.screen_size - 1,
        )
        y_out = (row * self.local_grid_size + center_offset).clamp(
            0,
            self.screen_size - 1,
        )
        return x_out.long(), y_out.long()


class CoarseToFineTargetHead(BaseSpatialTargetHead):
    head_type = "coarse_to_fine"

    def __init__(
        self,
        *,
        embed_dim: int,
        latent_dim: int,
        action_dim: int,
        coarse_grid_size: int,
        local_grid_size: int,
        screen_size: int,
        target_decode_mode: str = "center",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.coarse_grid_size = int(coarse_grid_size)
        self.local_grid_size = int(local_grid_size)
        self.screen_size = int(screen_size)
        self.target_decode_mode = str(target_decode_mode).lower()
        if self.target_decode_mode != "center":
            raise ValueError(
                f"Unsupported target_decode_mode: {target_decode_mode!r}",
            )
        if self.coarse_grid_size <= 0:
            raise ValueError("coarse_grid_size must be positive")
        if self.local_grid_size <= 0:
            raise ValueError("local_grid_size must be positive")
        if self.coarse_grid_size * self.local_grid_size != self.screen_size:
            raise ValueError(
                "Coarse-to-fine head expects coarse_grid_size * local_grid_size == screen_size, "
                f"got {self.coarse_grid_size} * {self.local_grid_size} != {self.screen_size}",
            )

        # Coarse stage: same as TokenPointer
        self.action_condition_embedding = nn.Embedding(int(action_dim), self.latent_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Fine stage: MLP that takes latent + action + coarse_token → fine_logits
        # We'll batch compute this for all coarse cells during build()
        self.fine_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2 + self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.local_grid_size * self.local_grid_size),
        )

    @property
    def token_count(self) -> int:
        return self.coarse_grid_size * self.coarse_grid_size

    @property
    def fine_count(self) -> int:
        return self.local_grid_size * self.local_grid_size

    def build(
        self,
        latent: torch.Tensor,
        spatial_context: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> TargetHeadState:
        batch_size, embed_dim, height, width = spatial_context.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Coarse-to-fine expected embed_dim={self.embed_dim}, got {embed_dim}",
            )
        if height != self.coarse_grid_size or width != self.coarse_grid_size:
            raise ValueError(
                "Coarse-to-fine grid mismatch: "
                f"expected {self.coarse_grid_size}x{self.coarse_grid_size}, got {height}x{width}",
            )
        action_ids = action_ids.clamp(
            0,
            self.action_condition_embedding.num_embeddings - 1,
        )
        action_emb = self.action_condition_embedding(action_ids)
        query = self.query_mlp(torch.cat((latent, action_emb), dim=-1))
        tokens = spatial_context.flatten(2).transpose(1, 2)
        coarse_logits = torch.einsum("bd,bnd->bn", query, self.token_proj(tokens))

        # Pre-compute fine logits for all coarse cells
        # tokens: [B, 49, embed_dim]
        # latent: [B, latent_dim]
        # action_emb: [B, latent_dim]
        coarse_tokens = tokens
        latent_expanded = latent.unsqueeze(1).expand(-1, self.token_count, -1)
        action_emb_expanded = action_emb.unsqueeze(1).expand(-1, self.token_count, -1)

        # fine_input: [B, 49, latent_dim*2 + embed_dim]
        fine_input = torch.cat((latent_expanded, action_emb_expanded, coarse_tokens), dim=-1)
        # fine_logits: [B, 49, 144]
        fine_logits = self.fine_mlp(fine_input).view(
            batch_size,
            self.token_count,
            self.fine_count,
        )

        return TargetHeadState(
            head_type=self.head_type,
            primary_logits=coarse_logits,  # [B, 49]
            secondary_logits=fine_logits,  # [B, 49, 144]
        )

    def sample(
        self,
        head_state: TargetHeadState,
        deterministic: bool = False,
    ) -> TargetSample:
        coarse_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        if deterministic:
            coarse_index = head_state.primary_logits.float().argmax(dim=-1)
        else:
            coarse_index = coarse_dist.sample()

        # Gather fine logits for the sampled coarse cells
        # head_state.secondary_logits: [B, 49, 144]
        batch_indices = torch.arange(
            coarse_index.size(0),
            device=coarse_index.device,
        )
        fine_logits = head_state.secondary_logits[batch_indices, coarse_index]  # [B, 144]

        fine_dist = torch.distributions.Categorical(logits=fine_logits.float())
        if deterministic:
            fine_index = fine_logits.argmax(dim=-1)
        else:
            fine_index = fine_dist.sample()

        x, y = self.decode_target_to_xy(coarse_index=coarse_index, fine_index=fine_index)
        coarse_entropy = self._normalized_entropy(
            coarse_dist.entropy(),
            head_state.primary_logits.size(-1),
        )
        fine_entropy = self._normalized_entropy(
            fine_dist.entropy(),
            fine_logits.size(-1),
        )

        return TargetSample(
            x=x,
            y=y,
            target_index=None,
            coarse_index=coarse_index,
            fine_index=fine_index,
            log_prob=coarse_dist.log_prob(coarse_index) + fine_dist.log_prob(fine_index),
            entropy=coarse_entropy + fine_entropy,
        )

    def evaluate(
        self,
        head_state: TargetHeadState,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> TargetEval:
        del target_index
        # Encode x, y to coarse, fine if not provided
        if coarse_index is None or fine_index is None:
            encoded = self.encode_xy_to_target(x.long(), y.long())
            if coarse_index is None:
                coarse_index = encoded["coarse_index"]
            if fine_index is None:
                fine_index = encoded["fine_index"]

        coarse_index = coarse_index.long()
        fine_index = fine_index.long()

        # Coarse evaluation
        coarse_dist = torch.distributions.Categorical(logits=head_state.primary_logits.float())
        coarse_log_prob = coarse_dist.log_prob(coarse_index)
        coarse_entropy = self._normalized_entropy(
            coarse_dist.entropy(),
            head_state.primary_logits.size(-1),
        )

        # Fine evaluation: gather logits for the RECORDED coarse cell (teacher-forcing)
        batch_indices = torch.arange(
            coarse_index.size(0),
            device=coarse_index.device,
        )
        fine_logits = head_state.secondary_logits[batch_indices, coarse_index]  # [B, 144]
        fine_dist = torch.distributions.Categorical(logits=fine_logits.float())
        fine_log_prob = fine_dist.log_prob(fine_index)
        fine_entropy = self._normalized_entropy(
            fine_dist.entropy(),
            fine_logits.size(-1),
        )

        return TargetEval(
            log_prob=coarse_log_prob + fine_log_prob,
            entropy=coarse_entropy + fine_entropy,
        )

    def encode_xy_to_target(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        x = x.long().clamp(0, self.screen_size - 1)
        y = y.long().clamp(0, self.screen_size - 1)

        # Coarse: 7x7 grid
        coarse_col = torch.div(x, self.local_grid_size, rounding_mode="floor")
        coarse_row = torch.div(y, self.local_grid_size, rounding_mode="floor")
        coarse_index = coarse_row * self.coarse_grid_size + coarse_col

        # Fine: 12x12 local grid
        fine_col = torch.remainder(x, self.local_grid_size)
        fine_row = torch.remainder(y, self.local_grid_size)
        fine_index = fine_row * self.local_grid_size + fine_col

        return {
            "x": x,
            "y": y,
            "target_index": None,
            "coarse_index": coarse_index.long(),
            "fine_index": fine_index.long(),
        }

    def decode_target_to_xy(
        self,
        *,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        target_index: torch.Tensor | None = None,
        coarse_index: torch.Tensor | None = None,
        fine_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del target_index
        if coarse_index is None or fine_index is None:
            raise ValueError("coarse_to_fine decode requires both coarse_index and fine_index")

        coarse_index = coarse_index.long().clamp(0, self.token_count - 1)
        fine_index = fine_index.long().clamp(0, self.fine_count - 1)

        # Decode coarse
        coarse_row = torch.div(coarse_index, self.coarse_grid_size, rounding_mode="floor")
        coarse_col = torch.remainder(coarse_index, self.coarse_grid_size)

        # Decode fine
        fine_row = torch.div(fine_index, self.local_grid_size, rounding_mode="floor")
        fine_col = torch.remainder(fine_index, self.local_grid_size)

        x_out = coarse_col * self.local_grid_size + fine_col
        y_out = coarse_row * self.local_grid_size + fine_row

        return x_out.long().clamp(0, self.screen_size - 1), y_out.long().clamp(0, self.screen_size - 1)
