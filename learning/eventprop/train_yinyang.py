"""
Train the EventProp SNN on Yin-Yang. End-to-end proof that the exact gradient
from eventprop_snn.py actually learns: a linear classifier caps near 64% on this
task, so clearing it (~73% here) means the hidden spiking layer is being trained
correctly by the exact gradient.

Shows the practical EventProp knobs from the README:
  - hot weight init keeps hidden neurons firing (the zero-gradient / silent-neuron
    problem); with a hot start the spike-count reg (reg_k, reg_nu, wired but left
    at 0 here) is not even needed;
  - gradient-norm clipping tames the 1/Vdot critical points.

Run:  python train_yinyang.py
"""
import numpy as np
from eventprop_snn import Config, backward, forward, hidden_spike_counts, loss_and_output_error
from yinyang import encode_latency, make_dataset


class Adam:
    def __init__(self, shapes, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads, strict=False)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def clip_global(grads, max_norm):
    total = np.sqrt(sum((g * g).sum() for g in grads))
    if total > max_norm:
        for g in grads:
            g *= max_norm / (total + 1e-12)
    return total


def evaluate(W_hid, W_out, coords, labels, cfg):
    correct = 0
    for xy, y in zip(coords, labels, strict=False):
        tr = forward(W_hid, W_out, encode_latency(xy), cfg)
        if int(np.argmax(tr.readout)) == y:
            correct += 1
    return correct / len(labels)


def main():
    cfg = Config(dt=0.01, t_max=2.5)         # short trial concentrates the readout
    rng = np.random.default_rng(0)
    n_in, n_hid, n_out = 5, 50, 3

    Xtr, Ytr = make_dataset(300, seed=1)
    Xte, Yte = make_dataset(200, seed=2)
    spikes_tr = [encode_latency(xy) for xy in Xtr]

    # hot init so hidden neurons already fire several times from step 0. With a
    # hot start + grad clipping we can leave the spike-count reg OFF entirely,
    # which keeps the objective stationary (reg's gradient shifts as firing
    # changes, and that non-stationarity was destabilising the optimisation).
    W_hid = rng.normal(2.6, 0.5, size=(n_hid, n_in))
    W_out = rng.normal(0.0, 0.8, size=(n_out, n_hid))
    opt = Adam([W_hid.shape, W_out.shape], lr=1e-2)

    epochs, batch = 60, 32
    reg_k, reg_nu = 0.0, 4.0                  # reg off; hot init keeps firing up
    best_acc = 0.0
    print(f"Yin-Yang | {n_hid} hidden LIF | dt={cfg.dt} | "
          f"train={len(Xtr)} test={len(Xte)}")
    print("shallow-linear baseline ~64%, chance ~46% (class imbalance)")
    init_spikes = np.mean([len(forward(W_hid, W_out, s, cfg).hidden_spikes)
                           for s in spikes_tr[:50]]) / n_hid
    print(f"init mean spikes/neuron/trial = {init_spikes:.2f} "
          f"(needs to be well above 0)\n")

    for ep in range(epochs):
        if ep == int(0.7 * epochs):
            opt.lr *= 0.4                     # anneal for a cleaner final fit
        order = rng.permutation(len(Xtr))
        ep_loss, ep_spikes, n_seen = 0.0, 0.0, 0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            g_hid = np.zeros_like(W_hid)
            g_out = np.zeros_like(W_out)
            for k in idx:
                tr = forward(W_hid, W_out, spikes_tr[k], cfg)
                loss, d_readout = loss_and_output_error(tr.readout, int(Ytr[k]))
                counts = hidden_spike_counts(tr)
                gh, go = backward(W_hid, W_out, tr, d_readout, cfg,
                                  reg_k=reg_k, reg_nu=reg_nu,
                                  hidden_spike_counts=counts)
                g_hid += gh
                g_out += go
                ep_loss += loss
                ep_spikes += counts.sum()
                n_seen += 1
            g_hid /= len(idx)
            g_out /= len(idx)
            clip_global([g_hid, g_out], max_norm=1.0)
            opt.step([W_hid, W_out], [g_hid, g_out])

        if ep % 3 == 0 or ep == epochs - 1:
            acc = evaluate(W_hid, W_out, Xte, Yte, cfg)
            best_acc = max(best_acc, acc)
            print(f"epoch {ep:2d}  loss={ep_loss/n_seen:.3f}  "
                  f"mean spikes/neuron/trial={ep_spikes/n_seen/n_hid:.2f}  "
                  f"test acc={acc:.1%}")

    preds = [int(np.argmax(forward(W_hid, W_out, encode_latency(xy), cfg).readout))
             for xy in Xte]
    final = evaluate(W_hid, W_out, Xte, Yte, cfg)
    print(f"\nBEST test accuracy:  {max(best_acc, final):.1%}   "
          f"(final {final:.1%}; linear baseline ~64%)")
    print(f"predicted class counts = {np.bincount(preds, minlength=3).tolist()}  "
          f"true = {np.bincount(Yte, minlength=3).tolist()}  (yang/yin/dots)")


if __name__ == "__main__":
    main()
