// Zone content for the SNN-PPO architecture explorer.
//
// ACCURACY CONTRACT: every shape, formula, and code excerpt below is
// pulled from THIS repository (verified 2026-07-06 against the files and
// line numbers cited). When the code and docs disagreed, the code won.
// Config values come from config.yaml (the live V7 run's settings).

export const PIPELINE_ZONES = [
  {
    id: 'pysc2',
    title: 'PySC2 Observation',
    subtitle: 'DefeatRoaches minigame, raw SC2 interface',
    color: '#38bdf8',
    geometry: 'globe',
    io: {
      in: [['SC2 game state', 'one env step (step_mul ticks)']],
      out: [
        ['feature_screen', '[27, 84, 84]'],
        ['feature_units', '[N, 23+] (all visible units)'],
        ['single/multi_select', '[K, 7]'],
        ['player', '[11]'],
        ['available_actions / last_actions', 'function-id lists'],
        ['score_cumulative', '[13]'],
      ],
    },
    math: `
      <p>No math here &mdash; this is the raw sensor boundary. Everything
      downstream is a function of these arrays:</p>
      <div class="eq">obs<sub>t</sub> = { screen, units, selection, player, available, last_actions, score }</div>`,
    why: `DefeatRoaches gives the agent a squad of marines against roaches on an 84×84 screen. The agent sees the same feature layers a human-oriented bot API exposes: per-pixel screen planes plus structured per-unit rows. Nothing is pre-digested — the whole point of the project is to learn the digestion. This is also where the step loop hands the observation to the extractor, together with the previous action token so the network can later judge what its last click actually did.`,
    code: {
      file: 'agent.py',
      lines: '251-255',
      text: `policy_input = self.extractor.extract_observation(
    obs,
    update_stats=not deterministic,
    last_action_token=self.last_action_token,
)`,
    },
  },
  {
    id: 'extractor',
    title: 'ObservationExtractor',
    subtitle: 'Running normalizers — obs_space/obs_space_2.py',
    color: '#22d3ee',
    geometry: 'filter',
    io: {
      in: [['PySC2 obs', 'raw arrays'], ['last_action_token', '[4]']],
      out: [['PolicyInputBatch', 'normalized, padded, masked tensors']],
    },
    math: `
      <p>Entity/selection features pass through a
      <b>RunningFeatureNormalizer</b> (Welford/Chan parallel update):</p>
      <div class="eq">&mu; &larr; &mu; + &Delta;&middot;(n<sub>b</sub>/n),&nbsp;&nbsp;
      M<sub>2</sub> &larr; M<sub>2</sub> + M<sub>2,b</sub> + &Delta;&sup2;&middot;n<sub>old</sub>n<sub>b</sub>/n</div>
      <div class="eq">x&#770; = clip( (x &minus; &mu;) / max(&sigma;, 0.01), &plusmn;10 )</div>
      <p>Normalization only activates after <b>count &ge; 32</b> samples
      (warm-up), and only for fields whose &sigma; is finite and above the
      floor.</p>`,
    why: `Unit health is in the hundreds, ratios are in [0,1], coordinates in [0,84) — without per-field standardization the big fields drown the small ones. The stats update online during rollouts and are FROZEN during deterministic eval (update_stats=not deterministic). The count and (mean, M2) arrays ship inside every checkpoint: an early Ray bug shipped count=0 normalizers, so eval silently fed the policy raw features it never trained on. That is why extractor state is now mandatory in every checkpoint and snapshot.`,
    code: {
      file: 'obs_space/obs_space_2.py',
      lines: '119-123',
      text: `delta = batch_mean - self.mean
total = self.count + batch_count
self.mean = self.mean + delta * (batch_count / total)
self.m2 = self.m2 + batch_m2 + (delta ** 2) * self.count * batch_count / total
self.count = total`,
    },
  },
  {
    id: 'batch',
    title: 'PolicyInputBatch',
    subtitle: 'Frozen input protocol — agent_core/policy_protocol.py',
    color: '#2dd4bf',
    geometry: 'crate',
    io: {
      in: [['extractor output', 'per-group tensors']],
      out: [
        ['spatial_obs', '[B, 27, 84, 84]'],
        ['entity_features + mask', '[B, 24, 21] + [B, 24]'],
        ['selection_features + mask', '[B, 20, 7] + [B, 20]'],
        ['action_feedback_tokens', '[B, 1, 12]'],
        ['meta_vec', '[B, 15]'],
        ['state_in (syn, mem)', '[B, 2, 95, 64] each, optional'],
      ],
    },
    math: `
      <p>The protocol is versioned and validated on every fragment:</p>
      <div class="eq">POLICY_PROTOCOL_VERSION = 3</div>
      <div class="eq">POLICY_INPUT_SCHEMA = "stream_action_effect_feedback_v2"</div>
      <p>meta_vec layout: player[0:11] &nbsp;|&nbsp; available-action bits[11:14]
      &nbsp;|&nbsp; last PySC2 action id[14:15].</p>`,
    why: `A frozen dataclass is the contract between the extractor, the network, the rollout actors, and every checkpoint on disk. Entity rows are padded to 24 slots and selection rows to 20, each with a boolean mask, because attention needs fixed shapes but the world has a variable unit count. The protocol version + schema string exist so a stale Ray actor or an old checkpoint fails loudly at validation instead of silently training on misaligned tensors.`,
    code: {
      file: 'agent_core/policy_protocol.py',
      lines: '206-213',
      text: `spatial_obs: torch.Tensor
entity_features: torch.Tensor
entity_mask: torch.Tensor
selection_features: torch.Tensor
selection_mask: torch.Tensor
meta_vec: torch.Tensor
state_in: SNNState | None = None
action_feedback_tokens: torch.Tensor | None = None`,
    },
  },
  {
    id: 'encoders',
    title: 'Per-Group Encoders',
    subtitle: 'CNN + entity/selection/feedback/meta MLPs + type embeddings',
    color: '#4ade80',
    geometry: 'prisms',
    io: {
      in: [['PolicyInputBatch tensors', 'per group']],
      out: [
        ['spatial tokens', '[B, 49, 64] (7×7 pooled + learned 2D pos)'],
        ['entity tokens', '[B, 24, 64]'],
        ['selection tokens', '[B, 20, 64]'],
        ['feedback token', '[B, 1, 64]'],
        ['meta token', '[B, 1, 64]'],
        ['fine_features (skip tap)', '[B, 32, 84, 84]'],
      ],
    },
    math: `
      <p>Spatial stack: conv1(27&rarr;16) &rarr; conv2(16&rarr;32) &rarr;
      <b>fine tap</b> &rarr; pool/2 &rarr; conv3(32&rarr;64) &rarr; pool/2 &rarr;
      AdaptiveAvgPool(7&times;7) &rarr; 49 tokens.</p>
      <p>Each spatial token gets a learned position code from its grid
      coordinate:</p>
      <div class="eq">tok<sub>i</sub> += MLP(2&rarr;64&rarr;64)( (x<sub>i</sub>, y<sub>i</sub>) &isin; [&minus;1,1]&sup2; )</div>
      <p>Every group then adds one of 5 learned token-type embeddings
      (SPATIAL=0, ENTITY=1, SELECTION=2, ACTION_FEEDBACK=3, META=4).</p>`,
    why: `Average-pooling the CNN map to 7×7 destroys sub-cell position on purpose — it buys a small token count — but the fine tap (taken just before the first pool, at full 84×84 resolution) preserves what the spatial head will later need to place a click inside a cell. Pooling is also permutation-blind, so the learned XY code is the only thing telling token 0 it is the top-left of the map. Type embeddings inject group identity once, up front, so attention can treat "who is speaking" as content.`,
    code: {
      file: 'agent_core/spiking_policy.py',
      lines: '851-863',
      text: `x = F.relu(self.conv1(spatial_input))
x = F.relu(self.conv2(x))
# Pre-pool per-pixel features for the fine skip connection; pooling
# below this line destroys sub-cell position, so this is the last
# tap with full spatial resolution.
fine_features = x if self._fine_skip_connection else None
x = self.pool(x)
x = F.relu(self.conv3(x))
x = self.pool(x)

spatial_tokens = self.token_pool(x)
spatial_tokens = spatial_tokens.flatten(2).transpose(1, 2)
spatial_tokens = self._add_spatial_positional_encoding(spatial_tokens)`,
    },
  },
  {
    id: 'stream',
    title: 'The 95-Token Stream',
    subtitle: '49 spatial + 24 entity + 20 selection + 1 feedback + 1 meta',
    color: '#a3e635',
    geometry: 'ring',
    io: {
      in: [['five token groups', 'each [B, n_g, 64]']],
      out: [['tokens', '[B, 95, 64]'], ['token_mask', '[B, 95] (bool)']],
    },
    math: `
      <p>Index layout (fixed):</p>
      <div class="eq">0&ndash;48 spatial &nbsp;|&nbsp; 49&ndash;72 entity &nbsp;|&nbsp;
      73&ndash;92 selection &nbsp;|&nbsp; 93 action-feedback &nbsp;|&nbsp; 94 meta</div>
      <p>The feedback token (dim 12) encodes what the LAST action did:
      bridge action type, normalized x/y, Smart-executed bit, score and
      kill-value deltas, target-near-enemy, moved-toward-target, and
      enemy/friendly health drops.</p>`,
    why: `One flat sequence lets a single attention block relate any unit to any map cell to the outcome of the previous click — no hand-wired pathways. The mask rides along so padded entity/selection slots contribute nothing anywhere downstream. Making action feedback a first-class TOKEN (rather than extra meta dims) was the v2-schema change: the outcome of your last click is information the network should attend to, and it is the signal the SIL trophy buffer later uses to verify a click actually engaged an enemy.`,
    code: {
      file: 'agent_core/spiking_policy.py',
      lines: '905-934',
      text: `tokens = torch.cat(
    (
        self._add_token_type(
            spatial_tokens,
            TOKEN_TYPE_SPATIAL,
            spatial_mask,
        ),
        self._add_token_type(
            entity_tokens,
            TOKEN_TYPE_ENTITY,
            entity_mask,
        ),
        self._add_token_type(
            selection_tokens,
            TOKEN_TYPE_SELECTION,
            selection_mask,
        ),
        self._add_token_type(
            action_feedback_tokens,
            TOKEN_TYPE_ACTION_FEEDBACK,
            action_feedback_mask,
        ),
        self._add_token_type(
            meta_tokens,
            TOKEN_TYPE_META,
            meta_mask,
        ),
    ),
    dim=1,
)`,
    },
  },
  {
    id: 'attention',
    title: 'Spiking Self-Attention',
    subtitle: 'Spikformer-style, binary-spike Q/K/V — stateless per step',
    color: '#f472b6',
    geometry: 'spikes',
    io: {
      in: [['tokens + mask', '[B, 95, 64] + [B, 95]']],
      out: [['attended tokens (+residual)', '[B, 95, 64]']],
    },
    math: `
      <p>Q/K/V are pushed through LIF neurons and become <b>binary
      spikes</b> before attention:</p>
      <div class="eq">spk = &Theta;(mem &minus; 1),&nbsp;&nbsp; mem&prime; = &beta;&middot;mem + Wx&nbsp;&nbsp;(snn.Leaky, &beta;<sub>qkv</sub>=0.5, learned)</div>
      <div class="eq">out = softmax&#8203;( spk(Q)&middot;spk(K)<sup>T</sup> / &radic;64 )&middot;spk(V) + residual</div>
      <p>Backward passes through the spike threshold via the
      <b>fast-sigmoid surrogate</b> gradient:</p>
      <div class="eq">&part;spk/&part;mem &asymp; 1 / (1 + k|mem &minus; 1|)&sup2;</div>`,
    why: `This block answers "what relates to what" within a single frame: spike-coded queries and keys make the attention matrix a count of coincident active features — cheap, sparse, and in keeping with the SNN theme — while the residual path keeps dense gradients flowing. It is deliberately STATELESS: its Q/K/V membranes are re-initialized every call, so all cross-step memory is owned by the next zone. The spike threshold is non-differentiable, so training substitutes the fast-sigmoid surrogate slope on the backward pass — the standard SNN trick that makes the whole network PPO-trainable.`,
    code: {
      file: 'agent_core/spiking_policy.py',
      lines: '286-310',
      text: `mem_q = self.lif_q.init_leaky()
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
).squeeze(1)`,
    },
  },
  {
    id: 'snn',
    title: 'Fast & Slow Token-Temporal SNN',
    subtitle: 'The real dual-timescale mechanism — state crosses env steps',
    color: '#c084fc',
    geometry: 'pathways',
    io: {
      in: [['attended tokens', '[B, 95, 64]'], ['state_in (syn, mem)', '[B, 2, 95, 64]']],
      out: [['combined spikes', '[B, 95, 64]'], ['state_out (syn, mem)', '[B, 2, 95, 64]']],
    },
    math: `
      <p>Each pathway is a synaptic LIF per token-dim (snn.Synaptic,
      &alpha;/&beta; <b>learned</b>, initialized from config):</p>
      <div class="eq">syn&prime; = &alpha;&middot;syn + I<sub>t</sub></div>
      <div class="eq">mem&prime; = &beta;&middot;mem + syn&prime; &minus; spk&middot;&theta;,&nbsp;&nbsp; spk = &Theta;(mem&prime; &minus; &theta;)</div>
      <div class="eq">fast: &alpha;=0.55, &beta;=0.65&nbsp;&nbsp;&nbsp;slow: &alpha;=0.92, &beta;=0.97</div>
      <div class="eq">out = (spk<sub>fast</sub> + spk<sub>slow</sub>) / 2&nbsp;&nbsp;(temporal_combine_mode: mean)</div>`,
    why: `This is where the agent remembers. The (syn, mem) tuple leaves the network, rides through the environment step, and comes back as state_in — the network is recurrent ACROSS frames, not within one. Two pathways give two integration horizons: the fast one (decay 0.55/0.65) reacts within a few frames — dodging, click outcomes — while the slow one (0.92/0.97) holds context for tens of frames — which flank the fight started on. Entity/selection state rows are deliberately zeroed each step (_zero_entity_state) because those slots are not identity-pinned yet: slot 3 may be a different roach next frame, and carrying its membrane would attach memory to the wrong unit.`,
    code: {
      file: 'agent_core/spiking_policy.py',
      lines: '952-968',
      text: `for _ in range(self.num_steps):
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
    syn_tok = torch.stack((fast_syn, slow_syn), dim=1) * pathway_token_mask_f
    mem_tok = torch.stack((fast_mem, slow_mem), dim=1) * pathway_token_mask_f`,
    },
  },
  {
    id: 'heads',
    title: 'Readout Heads',
    subtitle: 'Action head · coarse-to-fine spatial head · value head',
    color: '#fb923c',
    geometry: 'trident',
    io: {
      in: [
        ['latent', '[B, 64] (5-group masked-mean → 320 → 128 → 64)'],
        ['spatial_context', '[B, 64, 7, 7] (post-SNN spatial tokens)'],
        ['fine_features', '[B, 32, 84, 84] (skip tap)'],
      ],
      out: [
        ['action_logits', '[B, 3] (NO_OP / LEFT_CLICK / RIGHT_CLICK)'],
        ['coarse_logits', '[B, 49]'],
        ['fine_logits', '[B, 49, 144]'],
        ['value', '[B]'],
      ],
    },
    math: `
      <p>The click position is a <b>factorized categorical</b>: pick one of
      7&times;7 coarse cells, then one of 12&times;12 fine offsets inside it
      (49 &times; 144 = 7056 = 84&sup2; targets):</p>
      <div class="eq">log &pi;(x,y) = log p<sub>coarse</sub>(c) + log p<sub>fine</sub>(f | c)</div>
      <p>Coarse scores are a query&ndash;token dot product conditioned on the
      chosen action; fine logits get an additive <b>skip score</b> from the
      84&times;84 pre-pool features:</p>
      <div class="eq">fine += (q<sub>fine</sub> &middot; k<sub>pixel</sub>) / &radic;32</div>`,
    why: `A flat 7056-way softmax over pixels would be slow to learn and unstructured; the coarse×fine factorization matches how the spatial tokens are already organized. But the fine stage originally saw only the POOLED per-cell token — V5 diagnostics showed it emitting a constant sub-cell index (the dreaded corner-click), because nothing in its input varied within a cell. The fine skip connection fixes exactly that: per-pixel conv2 features are scored against a fine query, so where the click lands inside a cell can finally depend on the observation. LEFT_CLICK stays a scaffold: the dispatcher degrades it to no_op and the step is marked non-learnable.`,
    code: {
      file: 'agent_core/target_heads.py',
      lines: '522-548 (middle elided)',
      text: `action_emb = self.action_condition_embedding(action_ids)
query = self.query_mlp(torch.cat((latent, action_emb), dim=-1))
tokens = spatial_context.flatten(2).transpose(1, 2)
coarse_logits = torch.einsum("bd,bnd->bn", query, self.token_proj(tokens))
# ...
fine_logits = self.fine_mlp(fine_input).view(
    batch_size,
    self.token_count,
    self.fine_count,
)

if self.fine_skip_dim is not None:
    fine_logits = fine_logits + self._fine_skip_scores(
        fine_input,
        fine_features,
    )`,
    },
  },
  {
    id: 'dispatch',
    title: 'Action Dispatch',
    subtitle: 'ActionSpace → Smart_screen(x, y) or no_op',
    color: '#fbbf24',
    geometry: 'gate',
    io: {
      in: [['action_id, x, y', 'sampled from the heads']],
      out: [['PySC2 FunctionCall', 'Smart_screen("now", [x, y]) / no_op()']],
    },
    math: `
      <div class="eq">0 = NO_OP &rarr; no_op()</div>
      <div class="eq">1 = LEFT_CLICK &rarr; no_op()&nbsp;&nbsp;(scaffold; step marked learnable=False)</div>
      <div class="eq">2 = RIGHT_CLICK &rarr; Smart_screen("now", [x, y])</div>
      <p>Every dispatch also records a bridge token (type, x, y) that
      becomes part of the NEXT frame's action-feedback token.</p>`,
    why: `Smart_screen is SC2's right-click: attack-move onto enemies, move onto ground — one spatial verb that covers both engaging and kiting. Availability is checked against obs.available_actions before emitting, falling back to no_op if the function is not offered. One bootstrap select_army happens at episode start OUTSIDE PPO memory, so the policy never has to learn the selection ritual. The recorded bridge token closes the sensorimotor loop: next frame, the network sees what it just did — and the outcome detector sees whether that click touched an enemy.`,
    code: {
      file: 'action_space/action_space.py',
      lines: '62-67',
      text: `def right_click(self, obs, target_x, target_y, screen_size=None):
    target_x, target_y = self._clip_coords(target_x, target_y, screen_size)
    if actions.FUNCTIONS.Smart_screen.id in obs.observation.available_actions:
        self._set_token(BRIDGE_ACTION_RIGHT_CLICK, target_x, target_y, 0)
        return actions.FUNCTIONS.Smart_screen("now", [target_x, target_y])
    return self.no_op()`,
    },
  },
]

export const TRAINING_ZONES = [
  {
    id: 'fragments',
    title: 'Rollout Fragments',
    subtitle: 'RolloutFragment — the unit of experience',
    color: '#94a3b8',
    geometry: 'crate',
    io: {
      in: [['env steps from N Ray actors', 'fragment_steps ≈ 256 each']],
      out: [['RolloutFragment', '[T, ...] tensors + bootstrap tail + pre-step SNN state']],
    },
    math: `
      <p>Each fragment is self-contained: observations, actions, rewards,
      dones, the SNN state <i>before</i> each step, and a bootstrap tail
      (next policy input + state) so its GAE needs nothing from outside.
      Protocol version 3 / schema v2 are validated in __post_init__.</p>`,
    why: `Fragments are the seam that made Ray distribution a transport change instead of an algorithm change: the local trainer and the Ray learner consume the exact same objects through the same update_policy(fragments) path. Storing pre_step_snn_state per step is what lets TBPTT replay start a chunk from the true recurrent state it had during collection, and sample_mask carries which steps are learnable (helper/bootstrap steps are excluded).`,
    code: {
      file: 'distributed/protocol.py',
      lines: '85-111',
      text: `class RolloutFragment:
    actor_id: int
    fragment_id: int
    policy_version: int
    spatial_obs: torch.Tensor
    # ... entity/selection/feedback/meta tensors + masks ...
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    sample_mask: torch.Tensor
    pre_step_snn_state: SNNState | None
    tail_next_policy_input: PolicyInputBatch | None
    tail_next_snn_state: SNNState | None = None`,
    },
  },
  {
    id: 'gae',
    title: 'Per-Fragment GAE',
    subtitle: 'Advantage estimation with per-fragment bootstrap',
    color: '#38bdf8',
    geometry: 'filter',
    io: {
      in: [['rewards, values, dones', '[T] per fragment'], ['tail bootstrap value', 'V(s_T)']],
      out: [['advantages, returns', '[T]']],
    },
    math: `
      <div class="eq">&delta;<sub>t</sub> = r<sub>t</sub> + &gamma;&middot;V(s<sub>t+1</sub>)&middot;(1&minus;d<sub>t</sub>) &minus; V(s<sub>t</sub>)</div>
      <div class="eq">&Acirc;<sub>t</sub> = &delta;<sub>t</sub> + &gamma;&lambda;&middot;(1&minus;d<sub>t</sub>)&middot;&Acirc;<sub>t+1</sub></div>
      <div class="eq">&gamma; = 0.99, &lambda; = 0.95, returns = &Acirc; + V</div>
      <p>Time caps are stored as <b>truncations</b>, not terminal dones, so
      the value bootstrap continues through episode time-outs.</p>`,
    why: `With many actors contributing fragments, a global "last value" makes no sense — each fragment bootstraps from its own tail_next_policy_input, evaluated at update time. The truncation/termination distinction matters in DefeatRoaches because episodes routinely hit the step cap while marines are still fighting: treating that as death would teach the value function that time running out is catastrophic.`,
    code: {
      file: 'agent_core/ppo_trainer.py',
      lines: '2238-2247',
      text: `for t in reversed(range(rollout_size)):
    not_done = 1.0 - dones[t].float()
    delta = rewards[t] + gamma * next_value * not_done - values[t]
    running_advantage = (
        delta + gamma * gae_lambda * not_done * running_advantage
    )
    advantages[t] = running_advantage
    next_value = values[t]`,
    },
  },
  {
    id: 'tbptt',
    title: 'TBPTT Chunks',
    subtitle: 'Truncated backprop-through-time, window 128',
    color: '#22d3ee',
    geometry: 'ring',
    io: {
      in: [['ordered fragment steps', 'with stored pre-step SNN state']],
      out: [['chunks', '≤128 steps, split at episode boundaries']],
    },
    math: `
      <p>Chunks cut at <b>tbptt_window = 128</b> or at any done/reset,
      whichever comes first. Replay carries state forward inside a chunk:</p>
      <div class="eq">state &larr; chunk.initial_state;&nbsp; for t: out, state &larr; &pi;(obs<sub>t</sub>, state)</div>
      <p>Gradients flow through up to 128 steps of (syn, mem) dynamics and
      stop at chunk boundaries.</p>`,
    why: `Full backprop through a 2048-step rollout of a recurrent network is memory-impossible and gradient-useless; 128 steps is the compromise that still spans several engagements. Chunks must break at episode boundaries because recurrent state cannot legally flow across a reset. Chunks whose stored initial state came from an older policy version replay from that STORED state — replaying from zeros was one of the historical training bugs this design fixed.`,
    code: {
      file: 'agent_core/ppo_trainer.py',
      lines: '1501-1508',
      text: `window = rollout_size if self.tbptt_window is None else self.tbptt_window
start = 0
while start < rollout_size:
    end = min(start + window, rollout_size)
    split_mask = (dones[start:end] > 0.5) | episode_reset_mask[start:end]
    split_indices = torch.nonzero(split_mask, as_tuple=False)
    if len(split_indices) > 0:
        end = start + int(split_indices[0].item()) + 1`,
    },
  },
  {
    id: 'ppo',
    title: 'PPO Clipped Update',
    subtitle: '4 epochs, batch 2048, lr 5e-5, bf16 autocast',
    color: '#f472b6',
    geometry: 'spikes',
    io: {
      in: [['replayed logits/values', 'per chunk group'], ['advantages, old log-probs', '[T]']],
      out: [['gradient step', 'clip ε=0.10, target-KL 0.03 early stop']],
    },
    math: `
      <div class="eq">r<sub>t</sub> = exp(log&pi;<sub>new</sub> &minus; log&pi;<sub>old</sub>),&nbsp;&nbsp;log&pi; = log p(a) + [a spatial]&middot;log p(x,y)</div>
      <div class="eq">L<sup>CLIP</sup> = &minus;E[ min( r<sub>t</sub>&Acirc;<sub>t</sub>, clip(r<sub>t</sub>, 1&plusmn;0.10)&middot;&Acirc;<sub>t</sub> ) ]</div>
      <div class="eq">L = L<sup>CLIP</sup> + 0.5&middot;E[(R&minus;V)&sup2;] &minus; 0.01&middot;H&#771;</div>
      <p>Entropy H&#771; is normalized per head (action entropy / log 3 +
      spatial entropy) so 3-way and 7056-way heads contribute comparably.</p>`,
    why: `Standard PPO with two project-specific twists. First, the action log-prob and the spatial log-prob are summed only when the sampled action IS spatial — a no-op has no click to take credit for. Second, entropy is normalized per head; before that fix the 7056-way spatial head dominated the entropy bonus and the action head could collapse unnoticed. All losses are masked by sample_mask so bootstrap/helper steps never contribute gradient.`,
    code: {
      file: 'agent_core/ppo_trainer.py',
      lines: '2283-2288',
      text: `ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(
    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
) * advantages
policy_loss = -self._masked_mean(torch.min(surr1, surr2), sample_mask)`,
    },
  },
  {
    id: 'sil',
    title: 'SIL Trophy Buffer',
    subtitle: 'Self-imitation on verified-good clicks (V7, live)',
    color: '#fbbf24',
    geometry: 'gate',
    io: {
      in: [['verified single-step clicks', 'FIFO buffer, size 5000'], ['pre-step recurrent state', 'stored with each trophy']],
      out: [['auxiliary gradient step', 'sil_coef=0.5, after the PPO epochs']],
    },
    math: `
      <div class="eq">L<sup>SIL</sup> = &minus;0.5 &middot; E[ (R &minus; V(s))<sub>+</sub> &middot; log &pi;(a|s) ]</div>
      <p>The gate (R &minus; V)<sub>+</sub> is detached: a fixed per-sample
      weight, not a critic gradient. Admission requires the NEXT step's
      feedback token to confirm engagement (TARGET_NEAR_ENEMY or
      ENEMY_HEALTH_DROP) &mdash; return-gating alone was rejected.</p>`,
    why: `Rare good clicks were being washed out: one lucky engagement in 2048 steps contributes almost nothing to a PPO update. SIL (Oh et al. 2018) replays past good actions whose return still beats the current value estimate. The project twist is the admission rule — marine auto-attack inflates returns even for idle steps, so a trophy must be VERIFIED by the action-feedback token of the following frame: the click had to actually reach an enemy. Known open concerns: the extra optimizer step sits outside PPO's trust region, and stored recurrent states go stale as the network evolves.`,
    code: {
      file: 'agent_core/ppo_trainer.py',
      lines: '2505-2511',
      text: `# Gate = (R - V_current)+, detached: a fixed per-sample weight,
# not something we backprop through the critic.
weight = (returns_r - state_values.float()).clamp(min=0.0).detach()
sil_loss = -self.sil_coef * self._masked_mean(
    weight * new_log_prob,
    sample_mask,
)`,
    },
  },
]

// Token-stream composition, used by the 3D token ring and its legend.
export const TOKEN_GROUPS = [
  { name: 'spatial', count: 49, color: '#22d3ee' },
  { name: 'entity', count: 24, color: '#e879f9' },
  { name: 'selection', count: 20, color: '#4ade80' },
  { name: 'action_feedback', count: 1, color: '#fbbf24' },
  { name: 'meta', count: 1, color: '#f8fafc' },
]

// Time constants from config.yaml (initial values; learned during training).
export const SNN_TIME_CONSTANTS = {
  fast: { alpha: 0.55, beta: 0.65 },
  slow: { alpha: 0.92, beta: 0.97 },
}
