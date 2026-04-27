# GPU Throughput Optimization Plan

Created: 2026-04-27

This is the next-step optimization checklist for the Ray + fragment PPO path.
The goal is to make learner GPU time and rollout payload size visible before
we change the training algorithm into a more asynchronous shape.

## Current Instrumentation

The learner already logs enough to classify the first bottleneck:

| Metric | Meaning | First Read |
| --- | --- | --- |
| `rollout_wall_seconds` | wall time spent collecting rollout fragments | actor / SC2 throughput |
| `ray_get_wall_seconds` | time blocked waiting for Ray fragment refs | actor straggler / sync barrier |
| `update_wall_seconds` | total PPO learner update wall time | learner cost |
| `cpu_to_gpu_transfer_wall_seconds` | fragment and bootstrap tensors moved to learner device | transfer / payload issue |
| `chunk_pack_wall_seconds` | TBPTT chunks packed into time-major groups | packing overhead |
| `replay_forward_wall_seconds` | recurrent replay forward pass | main GPU forward cost |
| `backward_optimizer_wall_seconds` | backward, scaler, optimizer step | GPU training cost |
| `tbptt_forward_calls` | number of replay forward calls per update | batching quality |
| `tbptt_group_mean_active_chunks` | average active chunk rows per replay step | GPU occupancy proxy |
| `payload_total_mib` | rollout fragment payload size | Ray / transfer pressure |
| `cuda_peak_allocated_bytes` | peak active CUDA allocation | model + batch memory pressure |
| `cuda_peak_reserved_bytes` | peak CUDA allocator reservation | fragmentation / allocator pressure |
| `rollout_cache_spatial_dtype` | spatial rollout-cache dtype | payload and VRAM knob |

Quick command after a run:

```bash
python analyze_run.py --mode db --run-name <run_name>
python results.py --run-name <run_name> --report --aismart
```

## Diagnosis Matrix

### Actor-Bound

Symptoms:

- `rollout_wall_seconds` dominates total update cycle time
- `ray_get_wall_seconds` is close to `rollout_wall_seconds`
- `update_wall_seconds` is relatively small

Actions:

- Increase actor count until SC2/CPU saturates.
- Reduce straggler sensitivity with smaller `fragment_steps`.
- Move to pipelined sync collection before true async.
- Keep `serialize_env_resets: true` only if reset races are observed.

### Learner-Bound

Symptoms:

- `update_wall_seconds` dominates cycle time
- `replay_forward_wall_seconds` or `backward_optimizer_wall_seconds` dominates
- GPU utilization is high during update

Actions:

- Increase `batch_size` from `512` to `1024` if VRAM allows.
- Keep `epochs: 2-4` during throughput tuning.
- Watch `tbptt_group_mean_active_chunks`; values near `1.0` mean tiny GPU
  batches and poor occupancy.
- Prefer larger chunk groups over more PPO epochs when iteration speed matters.

### Transfer / Payload-Bound

Symptoms:

- `cpu_to_gpu_transfer_wall_seconds` is large
- `payload_total_mib` is high
- Ray object-store pressure or slow deserialization appears

Actions:

- Try `hyperparameters.rollout_cache_spatial_dtype: "float16"`.
- Keep scalar/index tensors compact in fragment transport.
- Avoid sending extractor state every weight broadcast; startup-only is usually enough.
- Consider compressing or quantizing `spatial_obs` transport if payload dominates.

### Packing-Bound

Symptoms:

- `chunk_pack_wall_seconds` is large relative to `update_wall_seconds`
- GPU is not busy during packing

Actions:

- Cache packed chunk buffers within an update where safe.
- Reduce raggedness by using fragment sizes that align with `tbptt_window`.
- Consider pre-packing fragments on CPU actors only if Ray payload growth is acceptable.

### Memory-Bound

Symptoms:

- `cuda_peak_allocated_bytes` approaches VRAM limit
- `cuda_peak_reserved_bytes` is much larger than allocated
- OOM happens when `batch_size` or actor count increases

Actions:

- Try spatial cache fp16 first.
- Reduce `batch_size` before reducing `tbptt_window`.
- Reduce `epochs` for tuning runs.
- Keep actor inference on CPU unless GPU has enough spare memory for rollout replicas.
- Track allocated vs reserved; high reserved-only pressure may be allocator fragmentation
  rather than true model footprint.

## Optimization Ladder

1. **Read one real timing row**
   - Use the metrics above from the current test run.
   - Do not optimize blind; choose the branch from the diagnosis matrix.

2. **Tune synchronous knobs**
   - `batch_size: 512 -> 1024`
   - `epochs: 4 -> 2` for fast iteration
   - `fragment_steps: 512` vs `256` / `1024`
   - `global_rollout_steps: 2048` vs `4096`
   - `rollout_cache_spatial_dtype: "float16"` if payload or VRAM is high

3. **Add pipelined sync**
   - Keep strict policy-version matching.
   - Submit next actor collection while the learner updates current fragments.
   - Consume only fragments from the current expected version.
   - This hides rollout time behind learner time without accepting stale PPO data.

4. **Add bounded async**
   - Add `max_policy_lag`, starting at `0`, then test `1`.
   - Log accepted, stale-dropped, and lagged fragment counts.
   - Reject anything older than `current_policy_version - max_policy_lag`.
   - Compare learning stability against sync and pipelined sync.

5. **Add support actors**
   - `LoggerActor` for single-writer SQLite.
   - `EvalActor` for deterministic eval and best-checkpoint promotion without blocking rollout.
   - Optional normalizer aggregation once actor-local extractor stats become a measured problem.

## First Decisions After The Current Run

Use this order:

1. If rollout dominates: tune actor count / fragment size, then pipelined sync.
2. If replay or backward dominates: tune `batch_size`, epochs, and chunk grouping.
3. If transfer dominates: try fp16 spatial cache and payload reduction.
4. If CUDA peak is high: lower batch size or fp16 cache before reducing TBPTT horizon.
5. If everything is balanced: implement pipelined sync as the next structural speedup.

## Guardrails

- Do not accept stale fragments until sync and pipelined sync have a baseline.
- Do not lower `tbptt_window` as the first fix; it changes credit assignment.
- Do not optimize only for steps/sec; watch reward, action mix, and eval traces.
- Keep checkpoint protocol compatibility explicit after any transport/layout change.
