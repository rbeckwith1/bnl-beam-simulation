"""
Batch-digitize AGS wall-current-monitor scope screenshots and fit the
binomial bunch-shape family  I(t) = I0 * [1-(t/T)^2]^J.

Graticule is auto-detected per image (the crops differ), so the only
per-file inputs are the voltage label and the on-screen dY readout.

Excluded: ScopeVrf207h6_3.png / ScopeVrf211h6_1.png are byte-identical
(md5 d574a12b677dccd4), so the voltage label of that trace is unknowable.

Outputs one CSV per trace plus a combined fit table.
"""
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


UP = r"C:\Users\rbeckwith\bnl-beam-simulation\src\tests\\"
OUT = r"C:\Users\rbeckwith\bnl-beam-simulation\src\tests\\"

# file, V_rf [kV], dY readout [mV] (baseline -> trough)
TRACES = [
    ("ScopeVrf207h6_1.png", 207.5, 690.7),
    ("ScopeVrf207h6_2.png", 207.5, 677.5),
    ("ScopeVrf211h6_2.png", 211.0, 652.2),
    ("ScopeVrf211h6_3.png", 211.0, 604.9),
]

NS_PER_DIV, FITW, J_LIST = 20.0, 45.0, (0.5, 1.5, 2.5)
COL = {0.5: "#0072B2", 1.5: "#D55E00", 2.5: "#009E73"}
DECIM = 2          # plot every Nth sample so markers stay distinguishable
 

NS_PER_DIV = 20.0
FITW = 45.0                       # fit window half-width [ns]
J_LIST = (0.5, 1.5, 2.5)



def find_graticule(a):
    H, W, _ = a.shape
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    grid = (G > 40) & (B > 40) & (R < 100) & ~((B > 150) & (G > 150))
 
    def cluster(idx):
        out, cur = [], [idx[0]]
        for v in idx[1:]:
            if v - cur[-1] <= 2:
                cur.append(v)
            else:
                out.append(int(np.mean(cur))); cur = [v]
        out.append(int(np.mean(cur)))
        return out
 
    def longest_uniform(p, tol=3):
        best = (0, 0)
        for i in range(len(p) - 1):
            for j in range(len(p) - 1, i + 1, -1):
                d = np.diff(p[i:j + 1])
                if d.max() - d.min() <= tol and (j - i) > (best[1] - best[0]):
                    best = (i, j); break
        return p[best[0]:best[1] + 1]
 
    return (longest_uniform(cluster(np.nonzero(grid.sum(0) > 0.25 * H)[0])),
            longest_uniform(cluster(np.nonzero(grid.sum(1) > 0.25 * W)[0])))
 
 
def digitize(path, dY):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    xs, ys = find_graticule(a)
    XL, XR, YT, YB = xs[0], xs[-1], ys[0], ys[-1]
    ndiv = int(round((XR - XL) / np.median(np.diff(xs))))
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    cy = (B > 150) & (G > 150) & (R < 120)
    cy[:YT, :] = False; cy[YB + 1:, :] = False
    yp = np.full(XR - XL + 1, np.nan)
    for i, x in enumerate(range(XL, XR + 1)):
        v = np.nonzero(cy[:, x])[0]
        if v.size:
            yp[i] = v.mean()
    ok = ~np.isnan(yp)
    yp = np.interp(np.arange(yp.size), np.nonzero(ok)[0], yp[ok])
    t = np.arange(yp.size) * NS_PER_DIV * ndiv / (XR - XL)
    ne = max(50, yp.size // 8)
    b = np.median(np.concatenate([yp[:ne], yp[-ne:]]))
    scale = dY / (yp.max() - b)
    I = (yp - b) * scale
    noise = np.std(np.concatenate([yp[:ne], yp[-ne:]]) - b) * scale
    return t, I, noise
 
 
def binom(t, I0, t0, T, J, b):
    return b + I0 * np.clip(1.0 - ((t - t0) / T) ** 2, 0.0, None) ** J
 
 
fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True,    #CHANGE 2 to 3 for LOG PLOT
                         gridspec_kw={"height_ratios": [2.6, 1.2], 
                                      "hspace": 0.16, "wspace": 0.22})
summary = []
 
for k, (fn, V, dY) in enumerate(TRACES):
    t, I, noise = digitize(UP + fn, dY)
    pk = t[np.argmax(I)]
    abv = t[I > 0.5 * I.max()]
    fwhm = abv[-1] - abv[0]
    m = np.abs(t - pk) < FITW
    tf, yf = t[m], I[m]
 
    fits = {}
    for J in J_LIST:
        T0 = 0.5 * fwhm / np.sqrt(1 - 2 ** (-1 / J))
        f = lambda x, I0, t0, T, b: binom(x, I0, t0, T, J, b)
        p, c = curve_fit(f, tf, yf, p0=[I.max(), pk, T0, 0.0], maxfev=60000)
        fits[J] = (p, np.sqrt(np.mean((yf - f(tf, *p)) ** 2)))
    pj, cj = curve_fit(binom, tf, yf, p0=[I.max(), pk, fwhm, 1.5, 0.0], maxfev=80000)
    ej = np.sqrt(np.diag(cj))
    rj = np.sqrt(np.mean((yf - binom(tf, *pj)) ** 2))
    summary.append((fn, V, fwhm, pj[2], ej[2], pj[3], ej[3],
                    pj[2] / np.sqrt(2 * pj[3] + 3), rj, noise, I.max(), fits))
 
    td, yd = (t - pk)[::DECIM], I[::DECIM]
    tt = np.linspace(-FITW, FITW, 3000) + pk
    axL, axR = axes[0, k], axes[1, k] #, axes[2, k] for LOG PLOTS and add back axG
 
    for ax in (axL,): #, axG):
        for J in J_LIST:
            p = fits[J][0]
            ax.plot(tt - pk, binom(tt, *p[:3], J, p[3]), color=COL[J],
                    lw=1.4, alpha=0.9, zorder=2,
                    label=f"J = {J}    T = {p[2]:.2f} ns")
        # ax.plot(tt - pk, binom(tt, *pj), color="k", lw=1.6, ls=(0, (5, 2)),
               # alpha=0.85, zorder=2,
                # label=f"J free = {pj[3]:.2f}   T = {pj[2]:.2f} ns")
        ax.plot(td, yd, "o", ms=3.4, mfc="none", mec="0.15", mew=0.7,
                alpha=0.85, zorder=3, label="digitized data")
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_xlim(-32, 32)
 
    axL.set_ylim(-0.07 * I.max(), 1.30 * I.max())
    axL.set_title(f"{fn.replace('.png','')}\n$V_{{rf}}$ = {V} kV\n"
                  f"FWHM = {fwhm:.2f} ns   |   peak = {I.max():.0f} mV",
                  fontsize=10, pad=6, linespacing=1.35)
    axL.legend(fontsize=7, framealpha=0.93, loc="upper right")
 
    # axG.set_yscale("log")
    # axG.set_ylim(1.2, 2.2 * I.max())
    # axG.axhspan(1.2, 3 * noise, color="0.75", alpha=0.45, zorder=0)
    # axG.text(0.03, 0.06, "noise floor", transform=axG.transAxes,
    #          fontsize=7.5, color="0.35")
    # axG.set_title("log scale — the wings are where J separates",
                  # fontsize=9, color="0.3")
 
    for J in J_LIST:
        p = fits[J][0]
        f = lambda x: binom(x, *p[:3], J, p[3])
        axR.plot(tf - pk, yf - f(tf), color=COL[J], lw=0.85)
    axR.plot(tf - pk, yf - binom(tf, *pj), "k", lw=0.9, ls=(0, (5, 2)), alpha=0.7)
    axR.axhspan(-noise, noise, color="0.6", alpha=0.35)
    axR.axhline(0, color="k", lw=0.5)
    axR.grid(alpha=0.22, lw=0.5)
    axR.set_xlabel("t − t$_{peak}$  [ns]", fontsize=10)
 
    if k == 0:
        axL.set_ylabel("beam current  [mV]", fontsize=10)
       # axG.set_ylabel("beam current  [mV, log]", fontsize=10)
        axR.set_ylabel("residual  [mV]", fontsize=10)
 
fig.suptitle("AGS wall current monitor — binomial bunch-shape fits  "
             "$I(t)=I_0\\,[1-(t/T)^2]^{\\,J}$", fontsize=14, y=0.995)
fig.savefig(OUT + "all_traces_binomial_fits.png", dpi=155, bbox_inches="tight")
 
print(f"{'trace':20s} {'V':>6} {'FWHM':>7} {'T_free':>14} {'J_free':>13} "
      f"{'sigma_t':>8} {'resid':>7} {'floor':>7}")
for (fn, V, fw, Tf, Te, Jf, Je, st, rj, nz, pkv, fits) in summary:
    print(f"{fn.replace('.png',''):20s} {V:>6.1f} {fw:>7.2f} "
          f"{Tf:>8.2f}±{Te:<5.2f} {Jf:>7.2f}±{Je:<5.2f} {st:>8.3f} "
          f"{100*rj/pkv:>6.2f}% {100*nz/pkv:>6.2f}%")
print("\nwrote all_traces_binomial_fits.png")
 