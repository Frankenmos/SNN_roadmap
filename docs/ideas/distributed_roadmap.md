# Distributed Scaling Roadmap: A Comparative Analysis

This document outlines two potential architectures for scaling our StarCraft II reinforcement learning agent: a Ray-based PPO setup and a custom Redis-based IMPALA setup.

## 1. Ray-based PPO Architecture

### a. Overview

Ray is a framework for building and running distributed applications. Its RLlib library is specifically designed for distributed reinforcement learning. In this architecture, we would continue to use PPO but leverage Ray to distribute the environment rollouts across multiple actors.

### b. Environment Wrapping

Wrapping our `DefeatRoaches` environment for Ray is straightforward. We would not need to manually create Ray Actors for each environment. Instead, we would pass our environment class directly to the RLlib trainer configuration. RLlib would then handle the instantiation of the environment in separate processes.

```python
# Example of RLlib configuration
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment(env="DefeatRoaches")
    .rollouts(num_rollout_workers=4)
)
```

### c. Shared Memory for Visual Observations

Ray is highly optimized for workloads with large numerical data, such as the visual observations in StarCraft II. It uses a shared-memory object store called **Plasma**. When an actor generates an observation, it is placed in the object store. The learner can then read this observation from shared memory without any costly deserialization or network transfer, as long as the learner is on the same node. This makes Ray particularly well-suited for our CPU-heavy environment, as it minimizes the data transfer overhead.

## 2. Custom Redis + IMPALA Architecture

### a. Overview

This approach involves building a custom distributed system using Redis as a message broker. We would switch from PPO to the IMPALA (Importance Weighted Actor-Learner Architecture), which is designed for asynchronous, off-policy training.

### b. Architecture

The system would consist of two main components:

*   **Learner (1, GPU-based):** This process is responsible for training the neural network. It continuously pulls batches of experience from a Redis-based replay buffer, computes gradients, and updates the model weights. It then publishes the new weights to Redis.
*   **Actors (4, CPU-based):** These processes are responsible for environment rollouts. Each actor runs an instance of the StarCraft II environment. It periodically pulls the latest model weights from Redis, generates a trajectory of experience (states, actions, rewards), and pushes it to the Redis replay buffer.

This architecture is highly scalable, but it requires more manual implementation effort compared to using Ray.

### c. V-Trace Algorithm

IMPALA's key innovation is the **V-trace algorithm**, which addresses the challenge of off-policy learning. In a distributed setup, actors are using a slightly older version of the policy than the learner. V-trace corrects for this "policy lag" by using importance sampling.

The V-trace target for the value function is calculated as follows:

$$
v_s = V(x_s) + \sum_{t=s}^{T-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t V
$$

where:
*   $V(x_s)$ is the value function at state $x_s$.
*   $\gamma$ is the discount factor.
*   $\delta_t V = \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))$ is a temporal difference error, scaled by an importance sampling ratio $\rho_t$.
*   $\rho_t = \min(\bar{\rho}, \frac{\pi(a_t|x_t)}{\mu(a_t|x_t)})$ is the importance sampling ratio, clipped at a maximum value $\bar{\rho}$. This clipping prevents the variance from exploding.
*   $c_i = \min(\bar{c}, \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)})$ is a similar clipped importance ratio used for the value target.

In essence, V-trace creates a new value target that is a blend of the actor's experience and the learner's current value function, allowing for stable off-policy updates.

## 3. Comparison and Recommendation

| Feature | Ray PPO | Redis IMPALA |
| :--- | :--- | :--- |
| **Development Effort** | Low | High |
| **Flexibility** | Medium | High |
| **Performance** | High (especially on a single machine) | Very High (scales across machines) |
| **Algorithm** | PPO (on-policy) | IMPALA (off-policy) |

For our specific use case, where the environment is CPU-heavy, the **Ray-based PPO architecture is the recommended starting point**. It offers a significant performance boost with minimal development effort. The shared memory system is a major advantage, and we can continue to use our existing PPO implementation.

If we find that we need to scale beyond a single machine or require more fine-grained control over the distributed architecture, we can then consider the Redis-based IMPALA approach.
