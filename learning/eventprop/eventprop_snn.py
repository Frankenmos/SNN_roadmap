"""
EventProp for a 1-hidden-layer spiking network — from scratch, in numpy.

This is Stage A of the EventProp learning ladder (see README.md): a *verified*
EventProp core, with nothing RL in it yet. The whole point is that the backward
pass here is the exact adjoint from Wunderlich & Pehle 2021, NOT a surrogate
gradient — and `gradient_check.py` proves it by matching finite differences.

Network
-------
    input spikes (latency coded)  ->  hidden LIF layer (spiking)  ->  output LI (non-spiking)

Neuron model (the canonical EventProp one): leaky integrate-and-fire with
*exponential-current synapses*. Each neuron has two state variables:

    tau_syn * I' = -I          (synaptic current, kicked by +w at each presyn spike)
    tau_mem * V' = -V + I      (membrane potential)

Hidden neurons spike when V >= v_th, then reset V -> 0 (their current I is not
touched by their own spike). Output neurons never spike; we read out the mean of
their membrane voltage over the trial and put a softmax cross-entropy loss on it
(this is ml_genn's `avg_var` readout — smooth, so the gradient is easy to check).

Everything is written for a *single sample* (vectorised over neurons, looped over
time). Batching is done by looping samples in train_yinyang.py — clarity over
speed; this is a teaching implementation.

How the backward pass works (and why it is "EventProp")
-------------------------------------------------------
EventProp computes the EXACT gradient of a spiking net by treating each spike as
a parameter-dependent event and running the adjoint backward through it. We get
the same exact gradient the robust way: reverse-mode (backprop) through the
discretised dynamics, with the spike time made a differentiable quantity by
linear interpolation of the threshold crossing.

The one idea that makes it EventProp rather than a surrogate gradient: a hidden
spike resets V to 0 (a constant), so there is NO gradient path through the
membrane value. The ONLY way a hidden weight influences the loss is by moving the
spike's crossing time `frac = (v_th - V_prev)/(V_new - V_prev)`. Differentiating
frac produces a 1/(V_new - V_prev) ~ 1/(dt * Vdot) factor -- exactly the
implicit-function-theorem 1/Vdot term from Wunderlich & Pehle. Where that factor
blows up (a near-tangential crossing) is the "critical point" where the loss is
genuinely discontinuous: EventProp's known blindness to spike creation/deletion,
showing up here as an O(dt) staircase in the loss.

Verification (gradient_check.py): in smooth regions the analytic gradient matches
finite differences to machine precision, across seeds and time constants. The
handful of weights sitting on a step-boundary crossing are skipped, because there
the discrete loss is genuinely non-differentiable (the critical points above).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    tau_mem: float = 2.0        # membrane time constant
    tau_syn: float = 0.5        # synaptic time constant
    v_th: float = 1.0           # spike threshold
    t_max: float = 4.0          # trial duration
    dt: float = 0.01            # integration step  -> n_steps = t_max/dt

    @property
    def n_steps(self) -> int:
        return int(round(self.t_max / self.dt))


@dataclass
class Trace:
    """Everything the backward pass needs from the forward pass."""
    readout: np.ndarray                       # (n_out,) mean output voltage
    v_out_series: np.ndarray                  # (n_steps, n_out)
    input_spike_step: np.ndarray              # (n_in,) step idx of input spike, -1=none
    hidden_spikes: list = field(default_factory=list)  # list of (step, j, frac, denom)
    n_in: int = 0
    n_hid: int = 0
    n_out: int = 0


def forward(W_hid: np.ndarray, W_out: np.ndarray,
            input_spike_times: np.ndarray, cfg: Config) -> Trace:
    """Simulate one trial. Returns a Trace (readout + info for backward).

    W_hid: (n_hid, n_in)   W_out: (n_out, n_hid)
    input_spike_times: (n_in,) spike time of each input neuron; np.inf = never.
    """
    n_hid, n_in = W_hid.shape
    n_out = W_out.shape[0]
    dt, tau_mem, tau_syn, v_th = cfg.dt, cfg.tau_mem, cfg.tau_syn, cfg.v_th
    n_steps = cfg.n_steps
    decay_syn = np.exp(-dt / tau_syn)

    # Pre-quantise input spikes to step indices.
    input_spike_step = np.full(n_in, -1, dtype=int)
    for i, t in enumerate(input_spike_times):
        if np.isfinite(t) and 0.0 <= t < cfg.t_max:
            input_spike_step[i] = int(t / dt)

    I_hid = np.zeros(n_hid)
    V_hid = np.zeros(n_hid)
    I_out = np.zeros(n_out)
    V_out = np.zeros(n_out)

    v_out_series = np.zeros((n_steps, n_out))
    hidden_spikes: list = []

    for t in range(n_steps):
        # --- hidden synaptic current: decay, then input kicks ---
        I_hid *= decay_syn
        fired_in = np.where(input_spike_step == t)[0]
        for i in fired_in:
            I_hid += W_hid[:, i]

        # --- hidden membrane integrate ---
        V_prev = V_hid.copy()
        V_hid = V_hid + (-V_hid + I_hid) / tau_mem * dt

        # --- hidden spikes (threshold on post-integrate V) ---
        crossed = np.where(V_hid >= v_th)[0]
        # output current: decay first (kicks from this step's hidden spikes below)
        I_out *= decay_syn
        for j in crossed:
            # Sub-step crossing fraction via linear interpolation of V across the
            # step. This makes the loss a SMOOTH function of the weights (the
            # spike time moves continuously), which is what lets finite
            # differences see the spike-timing gradient at all. Without it the
            # loss is a staircase in W_hid and FD reads ~0.
            denom = V_hid[j] - V_prev[j]          # = Vnew - Vprev > 0 at crossing
            frac = (v_th - V_prev[j]) / denom if denom > 1e-12 else 0.0
            frac = min(max(frac, 0.0), 1.0)
            hidden_spikes.append((t, int(j), float(frac), float(denom)))
            V_hid[j] = 0.0                       # zero reset
            # kick delivered at the interpolated crossing time, so it decays for
            # the remaining (1-frac) of this step before the step boundary.
            I_out += W_out[:, j] * np.exp(-(1.0 - frac) * dt / tau_syn)

        # --- output membrane integrate (non-spiking) ---
        V_out = V_out + (-V_out + I_out) / tau_mem * dt
        v_out_series[t] = V_out

    readout = v_out_series.mean(axis=0)          # avg_var readout
    return Trace(readout=readout, v_out_series=v_out_series,
                 input_spike_step=input_spike_step, hidden_spikes=hidden_spikes,
                 n_in=n_in, n_hid=n_hid, n_out=n_out)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def loss_and_output_error(readout: np.ndarray, label: int):
    """Softmax cross-entropy on the readout. Returns (loss, dL/dreadout)."""
    p = softmax(readout)
    loss = -np.log(p[label] + 1e-12)
    grad = p.copy()
    grad[label] -= 1.0                            # dL/dreadout = softmax - onehot
    return loss, grad


def backward(W_hid: np.ndarray, W_out: np.ndarray, trace: Trace,
             d_readout: np.ndarray, cfg: Config,
             reg_k: float = 0.0, reg_nu: float = 0.0,
             hidden_spike_counts: np.ndarray | None = None):
    """Exact gradient of the (smoothed) spiking forward, by reverse-mode adjoint.

    This is EventProp's content computed the robust way: reverse-mode through the
    discrete dynamics. The key EventProp insight lives in one place -- because a
    hidden spike resets V to 0 (a constant), the ONLY gradient path from a spike
    is through its interpolated crossing time `frac`, and d(frac)/dV carries a
    1/(Vnew-Vprev) ~ 1/(dt * Vdot) factor. That 1/Vdot is exactly the
    implicit-function-theorem term from the paper -- and where it blows up (a
    near-tangential crossing) is the "critical point" the loss-shaping paper warns
    about. Verified against finite differences in gradient_check.py.

    d_readout: (n_out,) = dL/dreadout (from loss_and_output_error, or an injected
               RL error signal later on -- that is the hook the actor-critic uses).
    reg_k, reg_nu, hidden_spike_counts: optional spike-count regularisation (the
               ml_genn / loss-shaping trick, see README). Blind in exact form, so
               it is injected as a heuristic learning signal at spikes. reg_k=0 off.

    Returns (grad_W_hid, grad_W_out).
    """
    n_hid, n_out = trace.n_hid, trace.n_out
    dt, tau_mem, tau_syn = cfg.dt, cfg.tau_mem, cfg.tau_syn
    n_steps = cfg.n_steps
    a = dt / tau_mem                       # membrane leak factor per step
    decay_syn = np.exp(-dt / tau_syn)
    drive = d_readout / n_steps            # dL/dV_out[t]: readout = mean over steps

    # Heuristic reg push toward target spike count (keeps firing neurons off
    # silence). Sign: below target (count<nu) -> negative push -> gradient descent
    # raises input weights -> more firing. Note it only acts on neurons that
    # already spike, so hot init still matters (see README).
    reg_push = np.zeros(n_hid)
    if reg_k > 0.0 and hidden_spike_counts is not None:
        reg_push = reg_k * (hidden_spike_counts - reg_nu)

    grad_W_hid = np.zeros_like(W_hid)
    grad_W_out = np.zeros_like(W_out)

    spikes_by_step: dict[int, list] = {}
    for (t, j, frac, denom) in trace.hidden_spikes:
        spikes_by_step.setdefault(t, []).append((j, frac, denom))
    inputs_by_step: dict[int, list] = {}
    for i, s in enumerate(trace.input_spike_step):
        if s >= 0:
            inputs_by_step.setdefault(int(s), []).append(i)

    # Output adjoints p_V,p_I = dL/dV_out, dL/dI_out (computed incrementally).
    pV_out = np.zeros(n_out)
    pI_out = np.zeros(n_out)
    # Hidden adjoints carried between steps: gV=dL/dV_hid[t], gI=dL/dI_hid[t].
    gV = np.zeros(n_hid)
    gI = np.zeros(n_hid)

    for t in range(n_steps - 1, -1, -1):
        # advance output adjoints to step t (exact discrete recursion)
        pV_out = drive + (1.0 - a) * pV_out
        pI_out = a * pV_out + decay_syn * pI_out

        # dL/dVnew for each hidden neuron. Non-spiking: the carried gV. Spiking:
        # V_hid[t]=0 kills the carried path; the spike-time path replaces it.
        dVnew = gV.copy()
        dVprev_extra = np.zeros(n_hid)
        for (j, frac, denom) in spikes_by_step.get(t, []):
            g = np.exp(-(1.0 - frac) * dt / tau_syn)
            dL_dg = W_out[:, j] @ pI_out               # spike kicks I_out by W_out*g
            dL_dfrac = dL_dg * g * (dt / tau_syn)
            grad_W_out[:, j] += pI_out * g
            dVnew[j] = dL_dfrac * (-frac / denom)      # d(frac)/dVnew, carried discarded
            dVprev_extra[j] = dL_dfrac * ((frac - 1.0) / denom)  # d(frac)/dVprev
            dVnew[j] += reg_push[j]                    # heuristic firing-rate push

        # dL/dI_hid[t] then input-weight grads at this step's input spikes
        gI_total = gI + a * dVnew
        for i in inputs_by_step.get(t, []):
            grad_W_hid[:, i] += gI_total

        # carry adjoints to step t-1
        gV = (1.0 - a) * dVnew + dVprev_extra
        gI = decay_syn * gI_total

    return grad_W_hid, grad_W_out


def hidden_spike_counts(trace: Trace) -> np.ndarray:
    counts = np.zeros(trace.n_hid)
    for (_, j, _, _) in trace.hidden_spikes:
        counts[j] += 1
    return counts
