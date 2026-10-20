"""The in-story API: what Luis types in his notebook.

`falkenrath.thermal` is Falkenrath-Voss's thermal library in the story. Here it
is a thin, honest wrapper around model.py, so every code cell shown on the page
really runs and really produces the numbers in the prose.
"""
import importlib.util
import os

import numpy as np

# Load model.py from this folder by path, so that a student's own `model.py`
# (or any other module of that name) can never be picked up by mistake.
_spec = importlib.util.spec_from_file_location(
    "_falkenrath_model", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py"))
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

class Stack:
    """A layered heat-shield stack. Lengths in mm, temperatures in K."""
    def __init__(self, name, t_adhesive, conductivity=None, k_of_x=None):
        self.name = name
        self._kfn = k_of_x            # optional k(x_mm) that varies inside a layer (graded)
        self._xs, self._ks = model.stack(t_adhesive)
        if conductivity is not None:                               # e.g. one material throughout
            self._ks = np.asarray(conductivity, float)
        # interfaces: where the conductivity actually changes (none for one material)
        self.interfaces = np.array([self._xs[i] for i in range(1, len(self._ks))
                                    if self._ks[i] != self._ks[i - 1]]) * model.L_STACK   # mm
        self.thickness = model.L_STACK
        self.adhesive = tuple(self._xs[1:3] * model.L_STACK)       # (hot face, cold face)
        self.adhesive_hot_face = self.adhesive[0]
        self.conductivity = tuple(float(k) for k in self._ks)     # per layer, hot side first
        self.T_hot, self.T_cold = model.T_HOT, model.T_COLD
    def k(self, x_mm):
        """Conductivity at depth x_mm (piecewise constant, layer by layer)."""
        if self._kfn is not None:
            return np.asarray(self._kfn(np.atleast_1d(np.asarray(x_mm, float))), float)
        x = np.atleast_1d(np.asarray(x_mm, float)) / model.L_STACK
        i = np.clip(np.searchsorted(self._xs, x, side="right") - 1, 0, len(self._ks) - 1)
        return self._ks[i]
    def heating(self, x_mm):
        """Heat absorbed per unit volume at depth x_mm, in the units that go
        with `conductivity`: -(k T')' = heating(x), x in mm, T in K."""
        L = model.L_STACK
        return (model.T_HOT - model.T_COLD) / L**2 * model.heating(np.asarray(x_mm) / L)
    def __repr__(self):
        return f"Stack({self.name!r}, {len(self._ks)} layers, {self.thickness:.0f} mm)"

class Solution:
    def __init__(self, stack, u, flux, label):
        self.stack, self._u, self._flux, self.label = stack, u, flux, label
    def at(self, x_mm):
        """Temperature in K at depth x_mm."""
        return float(model.to_kelvin(self._u(np.asarray(x_mm) / model.L_STACK)).ravel()[0])
    def temperature(self, x_mm):
        return model.to_kelvin(self._u(np.asarray(x_mm) / model.L_STACK))
    def flux(self, x_mm):
        """Heat flux, scaled to the flux at the hot face of the exact solution."""
        return np.asarray(self._flux(np.asarray(x_mm) / model.L_STACK))

class _Mara:
    """Mara's code: Galerkin, integrated Legendre basis (Lectures 16-17)."""
    START_DEGREE, TOLERANCE, MAX_DEGREE = 32, 1.0, 4096

    def solve(self, stack, degree=START_DEGREE):
        if stack._kfn is not None:
            raise NotImplementedError("Mara's code handles one conductivity per layer only")
        u, flux, _ = model.maras_code(stack._xs, stack._ks, degree)
        return Solution(stack, u, flux, f"Mara's code, degree {degree}")

    def certify(self, stack):
        """The certification workflow: start at degree 32 and keep doubling until
        two successive runs agree at the adhesive hot face to within 1 K.
        Returns (solution at the accepted degree, log of comparisons)."""
        x, n = stack.adhesive_hot_face, self.START_DEGREE
        prev = self.solve(stack, n)
        log = []
        while 2 * n <= self.MAX_DEGREE:
            cur = self.solve(stack, 2 * n)
            diff = abs(cur.at(x) - prev.at(x))
            log.append((n, prev.at(x), 2 * n, cur.at(x), diff))
            if diff < self.TOLERANCE:
                break
            n, prev = 2 * n, cur
        prev.degree = n
        return prev, log

class _Tent:
    """Tent: linear finite elements, a node at every interface."""
    def solve(self, stack, per_layer=1):
        if stack._kfn is not None:
            nodes, u = self._graded(stack, per_layer)
        else:
            nodes, u, _ = model.tent(stack._xs, stack._ks, per_layer)
        interp = lambda x: np.interp(x, nodes, u)
        return TentSolution(stack, interp, nodes * model.L_STACK, model.to_kelvin(u))

    @staticmethod
    def _graded(stack, per_layer):
        """Tent when k varies inside a layer: each element's matrix uses the
        integral of k over the element, (1/h^2) * int_e k dx * [[1, -1], [-1, 1]]."""
        from scipy.integrate import quad
        L, xs = model.L_STACK, stack._xs
        nodes = np.concatenate([np.linspace(xs[i], xs[i + 1], per_layer + 1)[:-1]
                                for i in range(len(xs) - 1)] + [[xs[-1]]])
        khat = lambda s: float(stack._kfn(np.array([s * L]))[0])
        ne = len(nodes) - 1
        A = np.zeros((ne + 1, ne + 1)); b = np.zeros(ne + 1)
        for e in range(ne):
            xa, xb = nodes[e], nodes[e + 1]; h = xb - xa
            A[e:e + 2, e:e + 2] += quad(khat, xa, xb)[0] / h**2 * np.array([[1, -1], [-1, 1]])
            b[e] += quad(lambda s: model.heating(s) * (xb - s) / h, xa, xb, epsabs=1e-14)[0]
            b[e + 1] += quad(lambda s: model.heating(s) * (s - xa) / h, xa, xb, epsabs=1e-14)[0]
        u = np.zeros(ne + 1); u[0], u[-1] = 1.0, 0.0
        u[1:-1] = np.linalg.solve(A[1:-1, 1:-1], b[1:-1] - A[1:-1, 0] * u[0] - A[1:-1, -1] * u[-1])
        return nodes, u

class TentSolution(Solution):
    def __init__(self, stack, u, nodes_mm, nodal_K):
        super().__init__(stack, u, None, "Tent")
        self.nodes, self.nodal = nodes_mm, nodal_K

def exact(stack):
    if stack._kfn is not None:
        raise NotImplementedError("no closed form for a graded stack; "
                                  "use tent.solve(stack, per_layer=1024) as a reference")
    u, flux = model.exact(stack._xs, stack._ks)
    return Solution(stack, u, flux, "exact")

mara = _Mara()
tent = _Tent()
KESTREL_B = Stack("Kestrel, stack B (thin bondline)", t_adhesive=0.01)
KESTREL_A = Stack("stack A (legacy bondline)", t_adhesive=0.1)
GRADED = Stack("graded tile (k rises through the tile)", t_adhesive=0.01,   # HW 6 only
               k_of_x=lambda x: np.where(x < 24.0, 1 + 2 * x / 24.0, np.where(x < 24.4, 0.3, 5.0)))
UNIFORM = Stack("uniform tile (no interfaces)", t_adhesive=0.1, conductivity=(1.0, 1.0, 1.0))  # HW 6 only
ADHESIVE_LIMIT = 440.0   # K  (the adhesive's limit, used throughout the story)
