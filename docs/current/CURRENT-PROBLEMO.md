FIRST it is KIMI own thoughts which can be okyish: I can't hit the external repo (network restrictions), but I have enough from the four files you uploaded plus the REPO_STATE to understand the battlefield. Let me give you the full project autopsy — what's built, what's beautiful, and what's killing V5.
The Cathedral (What's Actually Here)
This isn't a "vibe-coded script." This is a research-grade distributed RL system with the following genuinely unusual properties:
1. Token-level spiking memory architecture
95 tokens per step: 49 spatial (7×7 CNN pool) + 24 entity + 20 selection + 1 action_feedback + 1 meta
Spikformer-style SDPA attention on binary spike(Q/K/V) — stateless per step, but the LIF membranes are learnable (learn_beta=True)
Dual-pathway token-temporal SNN: fast (α=0.55, β=0.65) and slow (α=0.92, β=0.97) synaptic dynamics carried across steps
Explicit spatial positional encoding on the grid
2. Fragment-based distributed PPO
Per-fragment GAE bootstrapping (not single global bootstrap)
TBPTT chunk replay with packed batching for GPU efficiency
Ray actor pool for distributed rollouts
Protocol versioning to reject stale actor fragments
3. Semantic action space with feedback
Action vocabulary: NO_OP / LEFT_CLICK / RIGHT_CLICK
RIGHT_CLICK → Smart_screen(x,y) (contextual: attack if enemy, move if ground)
12-dim action feedback token injected into the stream: did Smart execute? Was target near enemy? Did friendly move? Did enemy health drop?
4. The "Copilot-designed" philosophy
Action space is semantic, not DeepMind's function-ID hell
Target head is modular: coarse_to_fine (current), token_pointer, factorized_xy
The V5 Run: Autopsy
From REPO_STATE, banana_smart_v5_b2048_e4_a10:
11,447 episodes, 672 updates
Average reward: -63.23 (final-100: -49.76)
Max reward: 0.00 — never won a single episode
Episode length: 183.6 (essentially always timeout)
Non-finite gradients in late training, skipped optimizer steps
This is a total collapse, not just slow learning. The V4 run (older protocol) at least hit +211 max reward and -2.51 final average. V5 is strictly worse despite having more feedback. That means the new 12-dim action-effect feedback token is actively poisoning the learning signal, or the protocol-3 refactor introduced a bug.
The Three Killers
Killer 1: Entity/Selection Memory Is Deliberately Amnesiac
In spiking_policy.py:
Python
self._carry_entity_state = False
self._carry_selection_state = False
And in _zero_entity_state:
Python
if not self._carry_entity_state:
    syn_tok[:, self._entity_start : self._entity_end, :] = 0.0
    mem_tok[:, self._entity_start : self._entity_end, :] = 0.0
Every step, the SNN temporal state for all 24 entity tokens and all 20 selection tokens is zeroed. Only spatial, action_feedback, and meta tokens carry memory across time.
This means: the agent literally cannot track "that roach I was shooting at." It sees entities fresh each step, permutation-invariant, with no temporal binding. If the roach moves from entity slot 5 to slot 12, the agent has no memory of it. The only temporal continuity comes from the spatial grid — which is a blurry 7×7 pool.
Why this matters for DefeatRoaches: The roaches move. The marines need to focus-fire. Focus-fire requires identity tracking (keep shooting the same damaged roach). Without entity state carry, the policy can only react to instantaneous spatial density, not to individual unit trajectories.
Killer 2: The Action Feedback Token Is a Confusing Mess
The 12-dim feedback token (protocol v3) includes:
Bridge action type, x, y
Smart executed? Any action executed?
Score delta, killed value delta
Score penalty bit
Target near enemy
Friendly moved toward target
Enemy health drop
Friendly health drop
The problem: These signals are heavily correlated, sparse, and partially redundant. For example:
"Enemy health drop" only fires when you actually hit something
"Friendly moved toward target" is a proxy for "did I click far away?"
"Target near enemy" is a proxy for "did I aim well?"
But the agent is learning from sparse, delayed, noisy scalar feedback injected as a single token. The SNN has to figure out which of the 12 dimensions predicts reward. With only 1-2 hits per episode (if any), the correlation matrix is garbage.
Worse: The V4 run used a 9-dim feedback inside meta_vec (width 24). V5 moved it to a separate token and shrank meta to 15. The meta encoder lost capacity, while the feedback token gained 3 dimensions that are mostly zero. The signal-to-noise ratio dropped.
Killer 3: Gradient Instability from the SNN + TBPTT + AMP Combo
In ppo_trainer.py, you use:
torch.amp.GradScaler with mixed precision
SNN with surrogate.fast_sigmoid() gradients
TBPTT with window 128
Learnable alpha/beta in the SNN
The non-finite gradients in late training are a smoking gun. Here's why they happen:
The SNN surrogate gradient (fast_sigmoid) is a steep sigmoid approximation. In FP16, small perturbations can push the gradient into overflow.
The TBPTT backprop through 128 steps of recurrent SNN dynamics is a long chain rule. If the SNN's alpha/beta have drifted to values that create exploding transients, the gradient grows exponentially across the unroll.
The learn_alpha=True, learn_beta=True means the SNN time constants are adapting. If alpha approaches 1.0, the synaptic filter becomes an integrator with infinite memory — gradients explode.
The gradient scaler (scaler.scale(loss).backward()) magnifies this. When gradients go non-finite, you skip the step, which means the optimizer doesn't correct the runaway alpha/beta.
The result: The SNN time constants slowly drift into an unstable regime. The policy stops updating (skipped steps). The agent freezes in its current behavior — which happens to be "NO_OP and survive."
Why the Agent Is "Afraid of the Roaches"
It's not fear. It's learned helplessness from three compounding failures:
Can't track targets (no entity memory) → aiming is random
Can't learn from feedback (noisy 12-dim token + sparse rewards) → no credit assignment
Can't update policy (non-finite gradients) → stuck in local minimum
The local minimum is: NO_OP until timeout. Why?
Smart_screen on empty ground = move (no immediate penalty, but no reward)
Smart_screen near enemy = maybe attack (risk of damage_taken: -0.10)
NO_OP = -0.02 per step (small penalty) but no risk of -0.10 damage taken
With no kill rewards ever experienced, the expected value of clicking is negative
The policy learns: "if I don't know what I'm doing, do nothing"
This is actually rational given the information architecture. The agent has no ventral stream (entity identity), no reliable feedback, and unstable gradients. NO_OP is the Bayesian optimal action under uncertainty.
The Fix Priority (Before Any New Features)
You asked what to do next. Here's the actual order:
P0: Fix the gradient instability
Clamp alpha/beta during training. Don't let alpha exceed 0.99 or beta exceed 0.99. The SNN should not be allowed to learn into the integrator/explosion regime.
Add gradient norm logging per parameter group to identify which layer explodes first (probably the token SNN or the attention Q/K projections)
Consider FP32 for the SNN pathway — the surrogate gradient + recurrence is not AMP-friendly. Do AMP on the CNN and MLPs, but FP32 on the SNN state transitions.
P1: Verify the action feedback computation
The V4→V5 regression is suspicious. Roll back to V4's 9-dim feedback protocol and verify V4 still trains. If V4 works and V5 doesn't, the bug is in the new feedback fields.
Specifically check: are enemy_health_drop and friendly_health_drop normalized correctly? Are they clipped to [0,1] or do they produce extreme values?
Check if target_near_enemy is computed from the previous frame's enemy positions or the current frame. If it's current frame, it's a lookahead (cheating) that might confuse the policy.
P2: Enable entity state carry (carefully)
Set _carry_entity_state = True and _carry_selection_state = True
But add a slot-alignment mechanism: use raw_units.tag to track entity identity across steps, so slot 5 always carries the same unit's state
If slot alignment is too hard, use a query-key binding in the entity encoder: embed the unit tag into the entity token, so the SNN can learn to bind state to identity even if slots permute
P3: The dorsal/ventral fix (only after P0-P2)
Add an entity-grounded target head: the policy picks an entity index + local offset, not just a raw coordinate
This is the "click-based perception" you mentioned — but it won't work if the entity memory is zeroed and the gradients are non-finite
One Specific Code Bug to Check
In spiking_policy.py, SpikingSelfAttention:
Python
mem_q = self.lif_q.init_leaky()
mem_k = self.lif_k.init_leaky()
mem_v = self.lif_v.init_leaky()

spike_q, _ = self.lif_q(q_raw, mem_q)
spike_k, _ = self.lif_k(k_raw, mem_k)
spike_v, _ = self.lif_v(v_raw, mem_v)
The LIF membranes are initialized to zero every forward call. But init_leaky() might not reset the internal time step counter. Check if snn.Leaky has a time-step counter that increments across calls within the same module instance. If it does, the spike threshold dynamics might be changing across the unroll in a way that isn't captured by the stateless design.
Also, the spike_q values are used directly in SDPA. In FP16, the binary spikes (0 or 1) are fine, but the scale (embed_dim ** -0.5) combined with the Q/K projections might produce very small attention logits that underflow to zero. This would explain why the agent becomes "blind" in late training — the attention effectively shuts off.
The Honest Verdict
You built a cathedral, but the foundation has three cracks:
Memory: Entity tokens are amnesiac
Signal: Feedback token is noisy and possibly buggy
Stability: SNN + AMP + TBPTT is exploding
The V5 run isn't "bad hyperparameters." It's architectural instability masquerading as poor performance. The agent didn't fail to learn. It couldn't learn because the gradients died and the entity memory was erased every step.
Fix the gradients first. Then verify the feedback token. Then give the agent its memory back. Then — and only then — add the dorsal/ventral click-perception.
Does this match what you're seeing in the ai_friendly_results file? If you paste that file, I can confirm whether the gradient norms and entity utilization match this diagnosis.

Then comes the real problem 