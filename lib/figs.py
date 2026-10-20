"""Figure styles for the book.

Two looks, because the story has two kinds of output:

* rough()     -- what a character sees in their own notebook at 2 a.m.:
                 plain matplotlib, gridlines, default-ish fonts.
* polished()  -- what a character sends to someone else (email attachments,
                 reports): the AE 370 house style from lectures/aefig.py.
* chalk()     -- drawings on Pruitt's board: light strokes on a transparent
                 background, for use inside `.board`.
* imagined()  -- what a character pictures in their head: no axes, no
                 numbers, soft translucent strokes. Used inside `.mindseye`.

Colour convention, shared with the lecture decks:
    BLUE   exact / true          ORANGE  the approximation
    GREY   structure, guides      RED     something that fails
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREY, RED = "#13294B", "#FF5F05", "#707372", "#B03A2E"
LAYER = "#C9CBCC"                     # interface lines and layer shading

_BASE = {"figure.dpi": 160, "savefig.bbox": "tight"}

_KEEP = ("backend", "interactive")      # resetting these breaks inline figures

def _reset():
    """Back to matplotlib defaults, keeping the backend and interactive mode
    that Quarto/Jupyter set up for inline figures."""
    mpl.rcParams.update({k: v for k, v in mpl.rcParamsDefault.items() if k not in _KEEP})

def rough():
    """A scratch-notebook look."""
    _reset()
    mpl.rcParams.update({**_BASE, "axes.grid": True, "grid.alpha": 0.35,
                         "font.size": 10, "lines.linewidth": 1.4,
                         "axes.prop_cycle": mpl.cycler(color=["#1f77b4", "#d62728", "#7f7f7f"])})

def polished():
    """The AE 370 house style."""
    _reset()
    mpl.rcParams.update({**_BASE, "font.size": 11, "axes.labelsize": 12,
                         "lines.linewidth": 2.2, "axes.edgecolor": GREY,
                         "axes.labelcolor": BLUE, "text.color": BLUE,
                         "xtick.color": GREY, "ytick.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, GREY, RED])})

def imagined():
    """A mind's-eye look: the shape of an idea, without axes or numbers."""
    _reset()
    mpl.rcParams.update({**_BASE, "savefig.transparent": True, "figure.facecolor": "none",
                         "axes.facecolor": "none", "lines.linewidth": 3.2,
                         "lines.solid_capstyle": "round", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "xtick.bottom": False, "ytick.left": False,
                         "xtick.labelbottom": False, "ytick.labelleft": False,
                         "text.color": GREY})

CHALK = "#e9efe9"

def chalk():
    """Board drawings: chalk-white strokes, no frame, transparent background."""
    _reset()
    mpl.rcParams.update({**_BASE, "savefig.transparent": True, "figure.facecolor": "none",
                         "axes.facecolor": "none", "lines.linewidth": 2.2,
                         "text.color": CHALK, "axes.edgecolor": CHALK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "xtick.bottom": False, "ytick.left": False,
                         "xtick.labelbottom": False, "ytick.labelleft": False,
                         "font.size": 13})

def layer_lines(ax, interfaces_mm, shade=None):
    """Vertical lines at each interface; optionally shade one layer (a, b) in mm."""
    for x in interfaces_mm:
        ax.axvline(x, color=LAYER, lw=1.0, zorder=0)
    if shade is not None:
        ax.axvspan(*shade, color=LAYER, alpha=0.35, lw=0, zorder=0)
