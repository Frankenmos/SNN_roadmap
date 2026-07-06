"""
The Yin-Yang dataset (Kriener, Goltz, Petrovici 2022) -- the standard tiny
benchmark for spike-timing SNNs. 2D points, 3 classes (yin / yang / dots), not
linearly separable, so a shallow classifier caps at ~64% and you need a hidden
layer. Perfect for checking that EventProp actually learns.

Input encoding matches the SNN literature: a point (x, y) becomes four values
(x, 1-x, y, 1-y) plus a bias, latency-coded into 5 input spikes (larger value ->
earlier spike). The (x,1-x,y,1-y) trick gives every input a strong spike
regardless of where the point sits, which keeps hidden neurons firing.
"""
import numpy as np

R_SMALL = 0.1
R_BIG = 0.5


def _which_class(x, y):
    d_right = np.sqrt((x - 1.5 * R_BIG) ** 2 + (y - R_BIG) ** 2)
    d_left = np.sqrt((x - 0.5 * R_BIG) ** 2 + (y - R_BIG) ** 2)
    crit1 = d_right <= R_SMALL
    crit2 = (d_left > R_SMALL) and (d_left <= 0.5 * R_BIG)
    crit3 = (y > R_BIG) and (d_right > 0.5 * R_BIG)
    is_yin = crit1 or crit2 or crit3
    is_dot = (d_right < R_SMALL) or (d_left < R_SMALL)
    if is_dot:
        return 2
    return int(is_yin)


def make_dataset(n, seed):
    """Return (coords[n,2] in [0,1]^2, labels[n]) sampled inside the big disc."""
    rng = np.random.default_rng(seed)
    coords, labels = [], []
    while len(coords) < n:
        x, y = rng.uniform(0.0, 2 * R_BIG, size=2)
        if (x - R_BIG) ** 2 + (y - R_BIG) ** 2 <= R_BIG ** 2:
            coords.append((x, y))
            labels.append(_which_class(x, y))
    return np.array(coords), np.array(labels, dtype=int)


def encode_latency(xy, t_early=0.1, t_late=1.6):
    """(x,y) -> 5 input spike times: (x, 1-x, y, 1-y) latency-coded + bias@0."""
    x, y = xy
    vals = np.array([x, 1.0 - x, y, 1.0 - y])
    spikes = t_early + (1.0 - vals) * (t_late - t_early)   # larger val -> earlier
    return np.array([spikes[0], spikes[1], spikes[2], spikes[3], 0.0])


if __name__ == "__main__":
    c, y = make_dataset(2000, 0)
    counts = np.bincount(y, minlength=3)
    print(f"2000 samples  class counts (yang/yin/dots) = {counts.tolist()}")
    print(f"example: xy={c[0].round(3).tolist()}  label={y[0]}  "
          f"spikes={encode_latency(c[0]).round(3).tolist()}")
