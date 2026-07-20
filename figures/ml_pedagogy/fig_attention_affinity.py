"""Pedagogical figure: a worked example of the self-attention operation.

This illustrates the arithmetic of the attention equation on three hits with
small, illustrative query/key/value vectors -- it shows what the operation
computes, not a result learned by the trained model. The flow is left to right:
the queries and keys give the scaled scores QK^T/sqrt(d_k); a row-wise softmax
turns each row into weights that sum to one; and the output of each hit is the
weighted sum of the value vectors, sum_j A_ij v_j. Hits 1 and 2 have aligned
queries and keys, so they attend to each other and their outputs blend; hit 3 is
different, attends to itself, and its output is essentially unchanged. This
blending of an input into a weighted average of the others is the whole point of
the operation.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from figstyle import PAL, save

# three hits; small illustrative query/key (d_k = 2) and value (d_v = 2) vectors.
# hits 1 and 2 point the same way in query/key space, hit 3 is orthogonal.
Q = np.array([[2.4, 0.4], [2.0, 0.8], [0.2, 2.6]])
K = np.array([[2.2, 0.2], [1.8, 0.6], [0.0, 2.4]])
V = np.array([[1.0, 3.0], [2.0, 2.0], [5.0, 0.5]])
d_k = Q.shape[1]

S = Q @ K.T / np.sqrt(d_k)                       # scaled scores
A = np.exp(S - S.max(1, keepdims=True))
A = A / A.sum(1, keepdims=True)                  # row softmax -> attention weights
O = A @ V                                        # contextualised output

hit = ["hit 1", "hit 2", "hit 3"]
REDS = LinearSegmentedColormap.from_list("hred", ["white", PAL["red"], "#7a0f16"])


def _txtcol(cmap, norm, v):
    r, g, b, _ = cmap(norm(v))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def numgrid(ax, M, *, cmap=None, norm=None, fmt="{:.2f}", ts=9.5,
            rlab=None, clab=None, title=None):
    """Draw a matrix as a grid of numbers, optionally colour-filled."""
    nr, nc = M.shape
    if cmap is None:
        ax.imshow(np.zeros((nr, nc)), cmap="Greys", vmin=0, vmax=1)
        cols = [["black"] * nc for _ in range(nr)]
    else:
        ax.imshow(M, cmap=cmap, norm=norm)
        cols = [[_txtcol(cmap, norm, M[i, j]) for j in range(nc)] for i in range(nr)]
    for i in range(nr):
        for j in range(nc):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=ts, color=cols[i][j])
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
    ax.grid(which="minor", color="0.7", lw=0.8)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(nc)); ax.set_yticks(range(nr))
    ax.set_xticklabels(clab if clab else [""] * nc, fontsize=8)
    ax.set_yticklabels(rlab if rlab else [""] * nr, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10.5, pad=6)


fig = plt.figure(figsize=(12.0, 4.6))
# axes placed by hand so the operation arrows can be drawn between them
axQ = fig.add_axes([0.035, 0.56, 0.085, 0.26])
axK = fig.add_axes([0.035, 0.20, 0.085, 0.26])
axV = fig.add_axes([0.035, 0.20, 0.085, 0.26])   # placeholder, moved below
axV.remove()
axS = fig.add_axes([0.26, 0.30, 0.135, 0.44])
axA = fig.add_axes([0.49, 0.30, 0.135, 0.44])
axVv = fig.add_axes([0.685, 0.60, 0.10, 0.24])
axO = fig.add_axes([0.86, 0.30, 0.115, 0.44])

numgrid(axQ, Q, rlab=hit, clab=["$q_1$", "$q_2$"], title="queries $\\mathbf{Q}$")
numgrid(axK, K, rlab=hit, clab=["$k_1$", "$k_2$"], title="keys $\\mathbf{K}$")

normS = mpl.colors.Normalize(0, S.max())
numgrid(axS, S, cmap=REDS, norm=normS, rlab=hit, clab=hit,
        title="scores  $\\mathbf{Q}\\mathbf{K}^{T}/\\sqrt{d_k}$")

normA = mpl.colors.Normalize(0, 1)
numgrid(axA, A, cmap=plt.cm.Blues, norm=normA, rlab=hit, clab=hit,
        title="weights  softmax")

numgrid(axVv, V, rlab=hit, clab=["$v_1$", "$v_2$"], title="values $\\mathbf{V}$")
numgrid(axO, O, rlab=hit, clab=["", ""],
        title="output $\\mathbf{A}\\mathbf{V}$")


def arrow(x0, y0, x1, y1, label, ly=0.045, below=False):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=16, lw=1.6, color="0.35",
                   shrinkA=2, shrinkB=2))
    if below:
        fig.text((x0 + x1) / 2, min(y0, y1) - ly, label, ha="center", va="top",
                 fontsize=9.5)
    else:
        fig.text((x0 + x1) / 2, max(y0, y1) + ly, label, ha="center", va="bottom",
                 fontsize=9.5)


arrow(0.135, 0.52, 0.238, 0.52, "$\\mathbf{Q}\\mathbf{K}^{T}/\\sqrt{d_k}$")
arrow(0.40, 0.52, 0.468, 0.52, "softmax\n(each row)", ly=0.02)
arrow(0.63, 0.52, 0.838, 0.52,
      "weighted sum   $\\sum_j A_{ij}\\,\\mathbf{v}_j$", ly=0.03, below=True)
# value vectors feed the final weighted sum
fig.add_artist(FancyArrowPatch((0.735, 0.60), (0.775, 0.53), transform=fig.transFigure,
               arrowstyle="-|>", mutation_scale=13, lw=1.4, color="0.55",
               connectionstyle="arc3,rad=-0.3", shrinkA=2, shrinkB=2))

# worked line for the first output row, using the real numbers
fig.text(0.5, 0.045,
         "row 1:  $\\mathbf{o}_1 = %.2f\\,\\mathbf{v}_1 + %.2f\\,\\mathbf{v}_2 + %.2f\\,\\mathbf{v}_3 "
         "= (%.2f,\\ %.2f)$   — hit 1 blends with hit 2 (its most-attended neighbour); "
         "hit 3 attends to itself, so $\\mathbf{o}_3 \\approx \\mathbf{v}_3$."
         % (A[0, 0], A[0, 1], A[0, 2], O[0, 0], O[0, 1]),
         ha="center", va="bottom", fontsize=9)
fig.text(0.5, 0.005,
         "Illustrative vectors showing the attention operation — not a result learned by the trained model.",
         ha="center", va="bottom", fontsize=8, style="italic", color="0.45")

paths = save(fig, "fig_attention_affinity")
print("scores:\n", np.round(S, 2))
print("attention rows (sum to 1):\n", np.round(A, 2))
print("outputs:\n", np.round(O, 2))
print("wrote:", *paths, sep="\n  ")
