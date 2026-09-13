# Phaseless Polynomial Interpolation

An intentionally simplified, educational implementation of phaseless polynomial
reconstruction over the rational numbers, accompanying:

**Michał R. Przybyłek and Paweł Siedlecki.**
*Information-Based Complexity vs Computational Complexity in Phaseless Polynomial
Interpolation.*
[arXiv:2603.21008](https://arxiv.org/abs/2603.21008), 2026.

The repository makes the educational implementation from the preprint's appendix
available as standalone Python code. Its purpose is to illustrate the
reconstruction strategy and support small, exact-arithmetic examples—not to
provide an optimized or production-ready solver. The paper is the reference for
the mathematical results and their complexity analysis.

## Problem and scope

Given pairwise distinct rational evaluation points $x_i$ and exact measurements
$y_i = |q(x_i)|$, the goal is to reconstruct polynomials
$q \in \mathbb{Q}[x]$ of degree at most $n$ satisfying

$$
q(x_i)^2 = y_i^2, \qquad i=1,\ldots,m.
$$

The polynomials $q$ and $-q$ have identical magnitudes, so reconstruction is
understood **up to a global sign**. Depending on the supplied measurements, there
may also be several solutions that are not related by a global sign.

This implementation illustrates the reconstruction method for

$$
m = 2n+1-k,
$$

where $k$ counts the missing measurements relative to $2n+1$. The fixed-$k$ setting
is the one relevant to the paper's polynomial-time reconstruction result; increasing
$k$ changes the computational regime.

The code takes evaluation points as input. It does **not** implement the paper's
adaptive selection of an additional evaluation point, its hardness reductions,
or a numerical method for noisy measurements.

## Installation

The module declares a dependency on **SymPy 1.12 or later** and uses Python 3.
The small examples in this README were checked with Python 3.13.5 and SymPy
1.14.0; this is not a full compatibility test matrix.

```bash
git clone https://github.com/micprzyb/phaseless-polynomial-interpolation.git
cd phaseless-polynomial-interpolation

python3 -m venv .venv
source .venv/bin/activate
python -m pip install "sympy>=1.12"
```

On Windows, create the environment with `py -m venv .venv` and activate it in
PowerShell with `.venv\Scripts\Activate.ps1` instead.

There is no package installation step: import the module from the repository
directory.

## Quick start

Run the following in a Python session started in the repository directory.
This example reconstructs $q(x)=x^2+1$ from four magnitudes. Here $n=2$ and $k=1$,
so the example exercises the Gröbner-basis branch of the implementation.

```python
import sympy as sp
from sympy.abc import x

from phaseless_interpolation import phaseless_interpolation

n = 2
points = [(0, 1), (1, 2), (2, 5), (3, 10)]
k = 2 * n + 1 - len(points)  # k = 1

candidates = phaseless_interpolation(points, k=k)

# Independently check every returned candidate using exact arithmetic.
assert candidates, "The implementation returned no candidate."
for q in candidates:
    assert sp.Poly(q, x, domain=sp.QQ).degree() <= n
    assert all(
        sp.expand(q.subs(x, xi)**2 - yi**2) == 0
        for xi, yi in points
    )

print([sp.expand(q) for q in candidates])
```

The final line prints:

```text
[x**2 + 1]
```

The function also prints diagnostic information, including the polynomial
system, Gröbner basis, and reconstructed candidates. The block above shows only
the final list, not the complete console output.

The assertions check the degree bound, rational coefficients, and agreement with
every supplied measurement. They do **not** establish that the returned list is
complete. See [Limitations](#limitations) before using the results elsewhere.

## Input and output

The public entry point in
[`phaseless_interpolation.py`](phaseless_interpolation.py) is
`phaseless_interpolation(points, k)`.

### Measurements

Supply a nonempty list of pairs `(xi, yi)`, using Python integers or exact SymPy
rational numbers. The evaluation points must be pairwise distinct. In the
phaseless formulation, `yi` is the nonnegative magnitude `abs(q(xi))`.

**Pass magnitudes, not squared magnitudes.** The function squares each `yi`
internally. Signed rational values are also accepted, but their signs are
ignored; they are not used as signed interpolation constraints.

Use `sp.Rational(1, 3)` rather than `1 / 3` or a decimal approximation. Python
evaluates `1 / 3` as a floating-point number before the interpolation function
receives it. See SymPy's
[notes on exact rational arithmetic](https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html#two-final-notes-and).

### Degree bound and `k`

There is no separate degree argument. For `m = len(points)`, the implementation
computes its degree bound as

```python
n = (m + k - 1) // 2
```

Choose your intended degree bound first, then set `k = 2*n + 1 - m`. To stay in
the setting of the reconstruction theorem, use an integer `k` with
$0 \leq k \leq n$ and require $m+k-1=2n$. In particular, $m \geq n+1$ and
$m+k-1$ must be even. These conditions are the caller's responsibility; the
implementation does not enforce them.

For example, degree bound $n=5$ corresponds to 10 measurements with `k=1`,
9 with `k=2`, or 8 with `k=3`. The reconstructed polynomial may have degree
strictly smaller than the bound.

### Returned candidates and sign convention

The return value is a list of SymPy expressions in the symbol `x` imported from
`sympy.abc`, not a list of coefficient arrays.

For nonzero data, the implementation selects the first nonzero measurement in
input order, shifts that evaluation point to the origin, and chooses the
positive square root there. Thus, a returned representative satisfies
$q(x_i)=|y_i|$ at that selected point; its negative is not separately returned.
Changing the order of the measurements can change the chosen representative's
sign.

Valid all-zero data return `[0]`. An empty list `[]` means that this
implementation returned no candidate; it must not be treated as a certified
proof of infeasibility, because some solver failures also produce `[]`.

### Example with genuine ambiguity

Four identical magnitudes at distinct nodes can be compatible with both a
constant and a nonconstant quadratic:

```python
import sympy as sp
from sympy.abc import x

from phaseless_interpolation import phaseless_interpolation

points = [(0, 1), (1, 1), (2, 1), (3, 1)]
candidates = phaseless_interpolation(points, k=1)

assert candidates
for q in candidates:
    assert sp.Poly(q, x, domain=sp.QQ).degree() <= 2
    assert all(
        sp.expand(q.subs(x, xi)**2 - yi**2) == 0
        for xi, yi in points
    )

print([sp.expand(q) for q in candidates])
```

The returned representatives are `1` and `x**2 - 3*x + 1`. They are not related
by a global sign. Compare results algebraically rather than relying on candidate
order or expression formatting.

## Reconstruction strategy

After shifting a nonzero measurement to the origin, write the shifted nodes as
$t_i$. Let $L(t)$ be the polynomial of degree less than $m$ interpolating the
squared measurements, and let

$$
R(t)=\prod_{i=1}^{m}(t-t_i).
$$

Every polynomial of degree at most $2n$ agreeing with those squared measurements
has the form

$$
P(t;\mathbf{c})
  = L(t) + R(t)\sum_{j=0}^{k-1} c_j t^j.
$$

The reconstruction problem is therefore to find parameters for which
$P(t;\mathbf{c})=Q(t)^2$ with $Q\in\mathbb{Q}[t]$ and $\deg Q\leq n$.
For $k=0$, the sum is empty and $P=L$.

The code fixes the positive constant coefficient of $Q$ at the shifted nonzero
measurement and computes the remaining coefficients recursively. For `k>0`, it
builds the remaining coefficient constraints, computes a lexicographic Gröbner
basis over `QQ`, and uses a simplified recursive triangular-system solver that
retains rational roots. Finally, it undoes the coordinate shift and removes
duplicate expressions.

The use of general-purpose symbolic routines, including `sympy.groebner` and
`sympy.solve`, keeps the implementation compact. The paper's complexity results
should not be read as practical runtime guarantees for these particular routines
or for this script.

## Running the bundled demonstrations

```bash
python phaseless_interpolation.py
```

The script's `__main__` block runs examples with `k=0`, `k=1`, `k=2`, and `k=3`,
including inconsistent measurements and the zero polynomial. These are printed
demonstrations, not an assertion-based test suite. Start with the small examples
above; the higher-degree symbolic systems can be substantially more expensive.

## Limitations

**Exact data only.** This is an algebraic demonstration over $\mathbb{Q}$.
Floating-point data, noisy measurements, and complex-valued phase retrieval are
outside its intended scope.

**Limited validation and error reporting.** Input shape, distinctness of nodes,
and the consistency of the degree/measurement parameters are not comprehensively
checked. Some exceptions during Gröbner-basis construction are caught and
reported only through an empty result.

**Candidates need independent verification.** In particular, the `k=0` branch
reconstructs coefficients without checking the full square identity or all
original measurements. Inconsistent data can therefore produce a spurious
candidate. The triangular-system solver is also deliberately simplified; this
repository should not be used as a certified general-purpose solver. Check every
candidate as in the examples above, and do not infer completeness merely from
successful checks.

**Symbolic cost can be high.** Polynomial expansion, coefficient growth, and
Gröbner-basis computation limit practical use. A polynomial-time result for each
fixed `k` is not a claim of uniformly efficient performance as `k` increases.

## Repository contents

```text
.
├── phaseless_interpolation.py  # Reconstruction routines and demonstrations
├── README.md                  # Scope, usage, and implementation notes
└── LICENSE                    # Apache License 2.0
```

## Citation and reproducibility

Please cite the accompanying paper when using this implementation in research:

```bibtex
@misc{przybylek_siedlecki_2026_phaseless,
  author        = {Przyby{\l}ek, Micha{\l} R. and Siedlecki, Pawe{\l}},
  title         = {Information-Based Complexity vs Computational Complexity
                   in Phaseless Polynomial Interpolation},
  year          = {2026},
  eprint        = {2603.21008},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CC},
  doi           = {10.48550/arXiv.2603.21008},
  url           = {https://arxiv.org/abs/2603.21008}
}
```

For reproducible use of the code, also record the repository commit or release,
Python and SymPy versions, input measurements, and `k`. From a cloned repository,
the following commands report the code revision and runtime versions:

```bash
git rev-parse HEAD
python --version
python -c "import sympy; print(sympy.__version__)"
```

## License

The repository includes the [Apache License, Version 2.0](LICENSE).
