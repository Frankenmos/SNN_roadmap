https://github.com/genn-team/ml_genn

# EventProp: exact gradients for spiking networks via the adjoint method

**EventProp computes exact loss gradients for spiking neural networks by treating spike events as parameter-dependent discontinuities in a hybrid dynamical system**, then applying the classical adjoint method with proper jump conditions at each spike time. Published by Timo C. Wunderlich and Christian Pehle in *Scientific Reports* (2021, arXiv:2009.08378), the method yields a remarkably simple gradient formula: the weight gradient is just a sum of adjoint variable values sampled at pre-synaptic spike times. Unlike surrogate gradient methods that smooth the spike threshold, EventProp produces gradients that match numerical finite-difference checks to **relative deviations below 10⁻⁷**. The backward pass is itself event-based — errors propagate only at spike times — making it **3× faster and 4× more memory-efficient** than surrogate-gradient BPTT in practice.

---

## The forward dynamics and what makes spikes hard

EventProp uses a network of **Leaky Integrate-and-Fire (LIF)** neurons with exponential synapses. Each neuron has two state variables — membrane potential *V* and synaptic current *I* — evolving between spikes as:

$$\tau_{\text{mem}} \dot{V} = -V + I, \qquad \tau_{\text{syn}} \dot{I} = -I$$

When neuron *n*'s membrane potential reaches threshold *ϑ*, a spike fires. The state undergoes an **instantaneous, parameter-dependent jump**: the membrane resets to zero (*V⁺ₙ = 0*), and post-synaptic currents receive weighted kicks (*I⁺ₘ = I⁻ₘ + wₘₙ*). This makes the SNN a **hybrid dynamical system** — smooth ODE flow punctuated by discrete state transitions whose timing depends on the parameters being optimized. Standard Neural ODE adjoint methods (Chen et al. 2018) assume purely continuous dynamics and cannot handle these parameter-dependent discontinuities directly.

The loss function combines spike-time-dependent terms and voltage-dependent terms: *L = lₚ(t^post) + ∫₀ᵀ lᵥ(V(t), t) dt*. In experiments, Wunderlich and Pehle used cross-entropy over first-spike latencies (Yin-Yang dataset, **98.1% accuracy**) and max-over-time voltage readout (MNIST, **97.6% accuracy**).

## How the adjoint method resolves the spike discontinuity

The derivation rests on two classical ideas combined in a novel way. First, the loss integral is **split at spike times** into continuous segments where standard calculus applies. Second, the **implicit function theorem** converts spike-time sensitivity into membrane-potential sensitivity via:

$$\frac{dt^{\text{post}}}{dw_{ji}} = -\frac{1}{\dot{V}^-_n} \cdot \frac{\partial V^-_n}{\partial w_{ji}}$$

This is valid whenever *V̇⁻ₙ ≠ 0* — the neuron must actually cross threshold, not merely graze it. The term **1/V̇⁻ₙ acts as a time-to-voltage conversion factor**, translating voltage-space errors into spike-timing errors. It diverges at "critical points" where spikes are created or destroyed (tangential threshold crossings), requiring gradient clipping in practice.

Between spikes, Lagrange multipliers λᵥ and λᵢ (the adjoint/costate variables) evolve backward in time according to their own leaky-integrator dynamics:

$$\tau_{\text{mem}} \lambda'_V = -\lambda_V - \partial l_V/\partial V, \qquad \tau_{\text{syn}} \lambda'_I = -\lambda_I + \lambda_V$$

where the prime denotes backward-time differentiation. These are initialized at *λᵥ(T) = λᵢ(T) = 0* and integrated from *T* back to *0*.

## Jump conditions: the mathematical heart of EventProp

At each spike time, the adjoint state of the spiking neuron undergoes a **discontinuous jump** that encodes how errors propagate backward through the spike event. For spiking neuron *n(k)* at time *tₖ*:

$$(\lambda_V^-)_{n} = \frac{\dot{V}^+_n}{\dot{V}^-_n} (\lambda_V^+)_n + \frac{1}{\tau_{\text{mem}} \dot{V}^-_n} \left[ \sum_{m \neq n} w_{nm}(\lambda_V^+ - \lambda_I^+)_m + \frac{\partial l_p}{\partial t_k} + l_V^- - l_V^+ \right]$$

Non-spiking neurons experience no jump in λᵥ, and **λᵢ never jumps** at any spike time. The term *Wᵀ(λᵥ⁺ − λᵢ⁺)* is the transposed-weight error routing familiar from standard backpropagation — it distributes error signals from post-synaptic to pre-synaptic neurons. The *∂lₚ/∂tₖ* term injects the spike-time loss gradient, while *lᵥ⁻ − lᵥ⁺* captures the discontinuity in the running voltage-dependent loss.

The final gradient expression is strikingly simple:

$$\frac{dL}{dw_{ji}} = -\tau_{\text{syn}} \sum_{\text{spikes from } i} (\lambda_I)_j$$

**Each weight gradient is just a weighted sum of the post-synaptic adjoint current, sampled at the pre-synaptic neuron's spike times.** Memory scales as O(S) where S is the total spike count — not the number of time steps — making EventProp especially efficient for sparse spiking regimes. No Dirac deltas appear in the final algorithm; the implicit function theorem resolves them into the well-defined jump conditions above.

## How EventProp relates to Neural ODEs and Pontryagin's principle

EventProp sits squarely in the **Pontryagin optimal control** tradition. The adjoint variables λ are precisely the costates of the Hamiltonian formulation, and the inter-spike adjoint ODE is the standard *λ̇ = ∂l/∂x − (∂F/∂x)ᵀλ* from classical optimal control. The gradient formula *dL/dpᵢ = −∫ λ · (∂F/∂pᵢ) dt* is the standard adjoint sensitivity result.

The **Neural ODE adjoint** (Chen et al. 2018) is a special case: it assumes purely continuous dynamics with no state jumps. EventProp extends this by incorporating the classical framework for **hybrid system sensitivity analysis** — specifically the work of Galán, Feehery & Barton (1999) and Serban & Recuero (2019) on parametric sensitivities in systems with discrete transitions. Where Neural ODEs solve one smooth adjoint ODE backward, EventProp solves a **piecewise adjoint ODE with event-based jumps** — the adjoint system itself becomes a "spiking" network that transmits error signals at discrete times. Chen, Amos & Nickel's later "Learning Neural Event Functions for ODEs" (ICLR 2021) brought event handling into the Neural ODE framework, explicitly citing EventProp and addressing similar mathematical challenges.

## Open-source implementations span multiple frameworks

The implementation landscape is fragmented across several ecosystems, each with different trade-offs:

- **`lolemacs/pytorch-eventprop`** (53 stars) — the most popular community implementation, pure PyTorch with Euler-discretized LIF dynamics and manual autograd backward pass. Limited to single-layer networks. Clean, readable code in `models.py` and `main.py`.

- **`genn-team/ml_genn`** with `EventPropCompiler` — the most feature-complete implementation, built on the GeNN GPU simulator by Thomas Nowotny's group at Sussex. Supports recurrent networks, learnable delays, heterogeneous time constants, convolutional layers, and deployment to **Intel Loihi 2**. Uses exponential Euler integration with hybrid time-driven/event-driven simulation.

- **`electronicvisions/jaxsnn`** (31 stars) — JAX-based, developed by Christian Pehle's group at Heidelberg. The only truly event-based implementation: it **analytically computes spike times** for LIF neurons rather than stepping through discrete time. Uses JAX's `custom_vjp` to define gradient rules through spike discontinuities. Interfaces with BrainScaleS-2 neuromorphic hardware. Install via `pip install jaxsnn`.

- **`tnowotny/genn_eventprop`** (16 stars) — research implementation accompanying the "loss shaping" paper. Jupyter notebook format, uses GeNN's sparse event-driven computation.

- **`mbalazs98/deventprop`** — extends EventProp to learn synaptic delays alongside weights (Mészáros et al., *Nature Communications* 2025). Built on mlGeNN. First event-based delay learning algorithm for recurrent SNNs.

- **`eventprop/eventprop`** — the official repository by Wunderlich, released late with only 3 stars. The community moved on to third-party implementations during the delay.

The official repo (`eventprop/eventprop`) and `hxtorch.snn` (PyTorch, for BrainScaleS-2 hardware-in-the-loop training) round out the options. **None of the major SNN frameworks (snnTorch, SpikingJelly, Spyx, Lava) include native EventProp support** — they focus on surrogate gradient methods.

## Follow-up work reveals both strengths and a fundamental limitation

The most significant extension is Nowotny, Turner & Knight's **"loss shaping"** paper (arXiv:2212.01232, published in *Neuromorphic Computing and Engineering*, 2025). It identified a fundamental limitation: **exact gradients provide no information about spike creation or deletion**, only about how existing spikes shift in time. Gradient descent can therefore inadvertently delete important spikes, increasing the loss. Loss shaping — designing loss functions that keep neurons near threshold — mitigates this problem and enabled EventProp to achieve **state-of-the-art results on the Spiking Heidelberg Digits benchmark**.

The delay learning extension by Mészáros et al. (*Nature Communications* 2025) is the bridge to **DCLS** (Hammouamri et al., ICLR 2024). DCLS learns synaptic delays using 1D temporal convolutions with learnable spacings in a discrete-time, surrogate-gradient framework. The two papers were developed independently — DCLS does not cite EventProp, and EventProp predates DCLS by three years. But Mészáros et al. directly compared the two approaches: EventProp-based delay learning uses **less than half the memory** and is **up to 26× faster** than DCLS, computes exact rather than surrogate gradients, and works on recurrent networks (DCLS is feedforward-only). The mathematical key: a delayed spike simply shifts when the post-synaptic current jump occurs, and the delay gradient is accumulated at saved spike times using time-shifted adjoint quantities.

EventProp has also been deployed on three neuromorphic platforms: **BrainScaleS-2** (Pehle et al., 2023), **SpiNNaker2** (Béna et al., NeurIPS 2024 Workshop), and **Loihi 2** (Shoesmith et al., 2025). The event-based backward pass maps naturally onto neuromorphic hardware — error signals use the same packet infrastructure as spikes.

## A four-session learning plan from variational principles to implementation

The resources below build a complete conceptual arc: *Lagrangian constraints → continuous adjoint → hybrid system jumps → EventProp*.

**Session 1 — Variational principles and the adjoint ODE (2–3 hours).** Start with the UCCS Pontryagin examples (Cascaval) for worked problems with complete solutions. Read the *Nature Communications Physics* beginner's guide to adjoint optimization (2024) for physical intuition. Then work through the Stanford adjoint tutorial by Andrew Bradley (cs.stanford.edu/~ambrad/adjoint_tutorial.pdf) sections 1–3 for the rigorous Lagrangian derivation. The Moritz Diehl PMP summer school slides provide the bridge from calculus of variations to Hamiltonian/costate formulation.

**Session 2 — Neural ODE adjoint method (2–3 hours).** The standout resource is Ilya Schurov's blog post "Adjoint State Method, Backpropagation and Neural ODEs" (ilya.schurov.com/post/adjoint-method/) — it builds from matrix multiplication efficiency through standard backprop to the continuous adjoint, showing they are the same idea. Follow with Vaibhav Patel's derivation using Lagrange multipliers specifically for Neural ODEs (vaipatel.com/posts/deriving-the-adjoint-equation-for-neural-odes-using-lagrange-multipliers/). Code along with the UvA Deep Learning tutorial notebook on dynamical systems and Neural ODEs. The **Depth First Learning: Neural ODEs curriculum** (depthfirstlearning.com/2019/NeuralODEs) provides an ideal multi-session structure with reading lists and exercises. For a demanding exercise, try the MIT 18.337 homework requiring from-scratch implementation in Julia (book.sciml.ai/homework/03/).

**Session 3 — Handling discontinuities and jump conditions (2–3 hours).** Chen, Amos & Nickel's "Learning Neural Event Functions for ODEs" (ICLR 2021, arXiv:2011.03902) is the key bridge paper — it extends Neural ODEs with learnable event functions and explicitly cites EventProp. Pair this with `torchdiffeq`'s `odeint_event` function for hands-on event handling. For the theoretical underpinning, study the adjoint jump conditions from Corner, Sandu & Sandu's hybrid multibody dynamics paper and Jia & Benson's Neural Jump SDEs (NeurIPS 2019), which show jump conditions in an ML context.

**Session 4 — SNN training and EventProp (3–4 hours).** Begin with Neftci, Mostafa & Zenke's surrogate gradient tutorial (IEEE Signal Processing Magazine, 2019) to understand what EventProp improves upon. Code a basic SNN using snnTorch tutorials 5–6 or Zenke's spytorch notebooks. Then read **Timo Wunderlich's own blog post** (timowunderlich.github.io/jekyll/update/2022/02/05/backprop.html) — the co-author's pedagogical explanation covering spike discontinuities, the adjoint method, and how they combine. Study the full paper (nature.com/articles/s41598-021-91786-z). Finally, implement a toy SNN using `lolemacs/pytorch-eventprop` as a reference, modifying the `SpikingLinear` module's `manual_backward` method to trace how the adjoint equations and jump conditions translate into code.

The conceptual thread connecting all four sessions is:

> *Constrained optimization via Lagrange multipliers produces the adjoint equation (Session 1) → Neural ODEs use this as continuous backpropagation (Session 2) → Hybrid systems require jump conditions at state discontinuities (Session 3) → LIF spikes are exactly such discontinuities, and EventProp's jump conditions yield exact backprop for SNNs (Session 4).*

## Conclusion

EventProp represents a clean application of **60-year-old optimal control theory** (Pontryagin 1962, Rozenvasser 1967) to a modern machine learning problem. Its mathematical elegance — exact gradients collapsing to a simple event-based sum — contrasts with the practical challenge that exact gradients are blind to spike creation and deletion. The loss shaping and delay learning extensions address these limitations, and neuromorphic hardware deployments demonstrate real-world viability. For someone building deep understanding, the derivation path from Lagrangian principles through Neural ODE adjoints to EventProp's spike-time jumps is one of the most instructive tours through the intersection of optimal control, dynamical systems, and deep learning. The key repositories for hands-on work are `lolemacs/pytorch-eventprop` (simplest entry point), `electronicvisions/jaxsnn` (most mathematically faithful, true event-based), and `genn-team/ml_genn` (most feature-complete, production-ready).