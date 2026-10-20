"""The story's physics and both solvers. Every number and figure in the book
comes from here, so prose and plots can't drift apart.

Problem: steady conduction through a layered heat-shield stack,
    -(k(x) T')' = q(x),   T(0) = T_hot,  T(L) = T_cold,
written nondimensionally on x in [0, 1] with u = (T - T_cold)/(T_hot - T_cold).

* Mara's code: Galerkin with the integrated-Legendre basis of Lectures 16-17
  (b_j = int_0^x P~_j), a linear lift for the boundary values, dense A.
* Tent: Galerkin with piecewise-linear hat functions, a node at every interface.
"""
import numpy as np
from numpy.polynomial import legendre as L
from scipy.integrate import quad

# ---------------------------------------------------------------- the stack
T_HOT, T_COLD = 1100.0, 300.0        # K: hot face, actively cooled face
L_STACK = 40.0                        # mm, total thickness
PRODUCTION_DEGREE = 32                # what the certification workflow runs

def stack(t_adhesive=0.01):
    """Interfaces (nondimensional) and conductivities: tile | adhesive | substrate."""
    xs = np.array([0.0, 0.6, 0.6 + t_adhesive, 1.0])
    ks = np.array([1.0, 0.3, 5.0])
    return xs, ks

Q0, DELTA = 8.0, 0.15                  # in-depth absorption: strength, penetration depth

def heating(x):                        # decays from the hot face
    return Q0 * np.exp(-x / DELTA)
def heating_integral(x):
    return Q0 * DELTA * (1.0 - np.exp(-x / DELTA))

def to_kelvin(u):
    return T_COLD + (T_HOT - T_COLD) * np.asarray(u)

# ------------------------------------------------------------ exact solution
def exact(xs, ks):
    """Exact u(x) and heat flux q(x) = -k u' for piecewise-constant k."""
    def u_of(x, C):
        tot = 0.0
        for i in range(len(ks)):
            b = min(xs[i + 1], x)
            if b > xs[i]:
                tot += quad(lambda s: (C - heating_integral(s)) / ks[i], xs[i], b,
                            epsabs=1e-13, epsrel=1e-13)[0]
            if x <= xs[i + 1]:
                break
        return 1.0 + tot
    a0 = u_of(1.0, 0.0) - 1.0
    a1 = u_of(1.0, 1.0) - 1.0 - a0
    C = (-1.0 - a0) / a1
    u = np.vectorize(lambda x: u_of(x, C))
    flux = lambda x: -(C - heating_integral(np.asarray(x)))
    return u, flux

# ------------------------------------------------------- Legendre machinery
def _legendre_table(n, t):
    """P_0..P_n evaluated at t in [-1,1], by the three-term recurrence."""
    P = np.empty((n + 1, t.size))
    P[0] = 1.0
    if n >= 1:
        P[1] = t
    for j in range(1, n):
        P[j + 1] = ((2 * j + 1) * t * P[j] - j * P[j - 1]) / (j + 1)
    return P

def _integrated(P, j):
    """b_j(x) = int_0^x P~_j for j >= 1, from P on t = 2x - 1."""
    return (P[j + 1] - P[j - 1]) / (2.0 * (2 * j + 1))

def _layer_quadrature(xs, ks, m):
    gx, gw = L.leggauss(m)
    X, W, K = [], [], []
    for i in range(len(ks)):
        a, b = xs[i], xs[i + 1]
        X.append(a + (b - a) * (gx + 1) / 2)
        W.append((b - a) / 2 * gw)
        K.append(np.full(m, ks[i]))
    return np.concatenate(X), np.concatenate(W), np.concatenate(K)

# ------------------------------------------------------------- Mara's code
def maras_code(xs, ks, n):
    """Lecture 16-17 Galerkin, degree n. Returns u_a(x) and its flux -k u_a'."""
    X, W, K = _layer_quadrature(xs, ks, n + 64)
    P = _legendre_table(n + 1, 2 * X - 1)              # P~_j(x) = P_j(2x-1)
    js = np.arange(1, n)
    Bd = P[js]                                          # b_j' = P~_j
    B = np.array([_integrated(P, j) for j in js])
    A = (Bd * K * W) @ Bd.T                             # a(b_j, b_i) = int k b_j' b_i'
    rhs = (B * W) @ heating(X) + (Bd * K * W).sum(axis=1)   # lift g = 1 - x, g' = -1
    c = np.linalg.solve(A, rhs)

    def u(x):
        x = np.atleast_1d(np.asarray(x, float))
        Px = _legendre_table(n + 1, 2 * x - 1)
        return 1 - x + sum(cj * _integrated(Px, j) for cj, j in zip(c, js))
    def flux(x):
        x = np.atleast_1d(np.asarray(x, float))
        Px = _legendre_table(n + 1, 2 * x - 1)
        k = ks[np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(ks) - 1)]
        return -k * (-1 + c @ Px[js])
    return u, flux, A

# ------------------------------------------------------------------- Tent
def tent(xs, ks, per_layer=1):
    """Linear finite elements, nodes at every interface, `per_layer` elements per layer."""
    nodes = np.concatenate([np.linspace(xs[i], xs[i + 1], per_layer + 1)[:-1]
                            for i in range(len(ks))] + [[xs[-1]]])
    kel = np.repeat(ks, per_layer)
    ne = len(nodes) - 1
    A = np.zeros((ne + 1, ne + 1)); b = np.zeros(ne + 1)
    for e in range(ne):
        xa, xb = nodes[e], nodes[e + 1]; h = xb - xa
        A[e:e + 2, e:e + 2] += kel[e] / h * np.array([[1, -1], [-1, 1]])
        b[e] += quad(lambda s: heating(s) * (xb - s) / h, xa, xb, epsabs=1e-14)[0]
        b[e + 1] += quad(lambda s: heating(s) * (s - xa) / h, xa, xb, epsabs=1e-14)[0]
    u = np.zeros(ne + 1); u[0], u[-1] = 1.0, 0.0
    rhs = b[1:-1] - A[1:-1, 0] * u[0] - A[1:-1, -1] * u[-1]
    u[1:-1] = np.linalg.solve(A[1:-1, 1:-1], rhs)
    return nodes, u, A
