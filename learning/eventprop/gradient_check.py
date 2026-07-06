"""
Finite-difference verification of the EventProp gradient.

This is the whole point of Stage A: prove the hand-written backward pass computes
the same gradient as numerically perturbing each weight and re-running the
forward. A *surrogate* gradient would NOT match finite differences here (it
approximates the threshold); EventProp does.

The one subtlety for a spiking net: the discretised loss has O(dt) "staircase"
steps wherever a spike sits on a timestep boundary. At those weights the loss is
genuinely non-differentiable (this is EventProp's spike creation/deletion
critical point), so central finite differences are meaningless there. We detect
them robustly: a weight is in a smooth region iff two finite-difference estimates
(at different eps) agree. We verify the gradient there and *report* how many were
skipped -- the skips are a feature, not a failure.

Run:  python gradient_check.py
"""
import numpy as np
from eventprop_snn import Config, backward, forward, loss_and_output_error

rng = np.random.default_rng(0)


def make_trial(seed, cfg, n_in=5, n_hid=8, n_out=3):
    r = np.random.default_rng(seed)
    # init hot enough to fire: a single kick w gives a peak PSP ~0.63*w for these
    # time constants, so w~1.6 reliably crosses v_th=1.
    W_hid = r.normal(1.8, 0.4, size=(n_hid, n_in))
    W_out = r.normal(0.0, 0.8, size=(n_out, n_hid))
    vals = r.uniform(0, 1, size=n_in)
    x = 0.1 + (1.0 - vals) * 1.5      # latency code: larger value -> earlier spike
    x[-1] = 0.0                       # last input is a bias, spikes at t=0
    label = int(r.integers(0, n_out))
    return W_hid, W_out, x, label


def verify(name, grad, loss_fn, params, cfg):
    """Compare analytic grad to finite differences, skipping non-smooth weights."""
    kept_a, kept_f, skipped, zero = [], [], 0, 0
    it = np.ndindex(*grad.shape)
    for idx in it:
        w0 = params[idx]

        def L(w, idx=idx):
            p = params.copy()
            p[idx] = w
            return loss_fn(p)

        fd1 = (L(w0 + 1e-3) - L(w0 - 1e-3)) / 2e-3
        fd2 = (L(w0 + 5e-4) - L(w0 - 5e-4)) / 1e-3
        if abs(fd1) < 2e-4 and abs(fd2) < 2e-4:
            zero += 1
            continue
        if abs(fd1 - fd2) > 0.05 * max(abs(fd1), abs(fd2)):
            skipped += 1                       # staircase step near w0 -> FD unreliable
            continue
        kept_a.append(grad[idx])
        kept_f.append(fd1)
    a, f = np.array(kept_a), np.array(kept_f)
    cos = (a @ f) / (np.linalg.norm(a) * np.linalg.norm(f) + 1e-12)
    rel = np.linalg.norm(a - f) / (np.linalg.norm(f) + 1e-12)
    ok = "OK " if (cos > 0.9999 and rel < 0.01) else "!! "
    print(f"  {ok}{name:6}  checked={len(a):2d}  skipped(critical pts)={skipped:2d}"
          f"  near-zero={zero:2d}   cos={cos:.6f}  relL2={rel:.4%}")
    return cos > 0.9999 and rel < 0.01


def main():
    cfg = Config(dt=0.0025, t_max=4.0)
    print(f"EventProp gradient check  (dt={cfg.dt}, tau_mem={cfg.tau_mem}, "
          f"tau_syn={cfg.tau_syn})\n")
    all_ok = True
    for seed in range(5):
        W_hid, W_out, x, label = make_trial(seed, cfg)
        tr = forward(W_hid, W_out, x, cfg)
        loss, d_readout = loss_and_output_error(tr.readout, label)
        g_hid, g_out = backward(W_hid, W_out, tr, d_readout, cfg)
        print(f"seed {seed}:  loss={loss:.4f}  hidden spikes={len(tr.hidden_spikes)}")

        def loss_hid(p, W_out=W_out, x=x, label=label):
            t = forward(p, W_out, x, cfg)
            return loss_and_output_error(t.readout, label)[0]

        def loss_out(p, W_hid=W_hid, x=x, label=label):
            t = forward(W_hid, p, x, cfg)
            return loss_and_output_error(t.readout, label)[0]

        all_ok &= verify("W_out", g_out, loss_out, W_out, cfg)
        all_ok &= verify("W_hid", g_hid, loss_hid, W_hid, cfg)
    print("\n" + ("ALL CHECKS PASSED -- the adjoint is exact." if all_ok
                  else "SOME CHECKS FAILED."))


if __name__ == "__main__":
    main()
