# Project Overview: PySC2 Spiking PPO Agent (DefeatRoaches)

This project implements a **modular reinforcement learning stack** on top of **PySC2**, targeting the `DefeatRoaches` mini-game.  
It combines:

- a **spiking neural network (SNN)** policy (`PolicyNetwork`) built with **PyTorch + snntorch**
- a **PPO agent** (`DefeatRoaches`) with custom observation, action, and reward wrappers
- a **modern Python environment** (3.10–3.12) that still supports PySC2 via a light `protobuf` pin.

The goal is to have a clean playground for **experiments on SNN-based RL** in StarCraft II, with a codebase that’s modular and debuggable.

---

## 1. Environment Setup

### 1.1. Python & Virtual Environment

Use **Python 3.10–3.12**. Example with conda:

```bash
conda create -n sc2_snn_ppo python=3.12
conda activate sc2_snn_ppo
