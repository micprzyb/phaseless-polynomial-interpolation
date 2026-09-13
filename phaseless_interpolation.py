"""
PHASELESS INTERPOLATION ALGORITHM FOR RATIONAL POLYNOMIALS
================================================================================
Description:
    An instructional, highly non-efficient, but polynomial-time, implementation
    of phaseless interpolation for rational polynomials of degree n given 2n+1-k
    for some (presumably fixed) parameter k. The algorithm runs in polynomial-time
    in the bit-size of the input points and doubly-exponential in k.
    The algorithm is described in:
        "Information-Based Complexity vs Computational Complexity in Phaseless Polynomial
Interpolation"
            by Michal R. Przybylek and Pawel Siedlecki


Usage:
    python phaseless_interpolation.py

Requirements:
    sympy>=1.12
"""

__author__     = "Michal R. Przybylek"
__license__    = "MIT"
__version__    = "1.0.0"
__email__      = "mrp@mimuw.edu.pl"



import sympy as sp
from sympy.abc import x


# ==========================================
# Part 1: Triangular System Solver
# ==========================================

def _simplify_polys(polys, partial_solution):
    """Substitutes partial solutions and expands polynomials."""
    current_polys = [sp.expand(p.subs(partial_solution)) for p in polys]
    return [p for p in current_polys if p != 0]


def _check_inconsistency(active_polys):
    """Returns True if any polynomial simplifies to a non-zero number."""
    return any(p.is_Number and p != 0 for p in active_polys)


def _find_target_poly(active_polys, target_var):
    """Identifies a univariate polynomial for the target variable."""
    for p in active_polys:
        syms = p.free_symbols
        if not syms: continue
        if syms == {target_var} or syms.issubset({target_var}):
            return p
    return None


def _find_rational_roots(poly, target_var):
    """Solves a univariate polynomial and returns strictly rational roots."""
    roots = sp.solve(poly, target_var, dict=True)
    valid_roots = []
    for root in roots:
        val = root[target_var]
        if val.is_number and val.is_rational:
            valid_roots.append(val)
    return valid_roots


def solve_triangular_system(polys, variables, partial_solution=None):
    """Recursively solves a triangular system of polynomials."""
    if partial_solution is None: partial_solution = {}
    if not variables: return [partial_solution]

    target_var, remaining = variables[-1], variables[:-1]
    active_polys = _simplify_polys(polys, partial_solution)

    if _check_inconsistency(active_polys): return []

    uni_poly = _find_target_poly(active_polys, target_var)
    if uni_poly is None: return []

    full_solutions = []
    for val in _find_rational_roots(uni_poly, target_var):
        new_partial = partial_solution.copy()
        new_partial[target_var] = val
        full_solutions.extend(solve_triangular_system(polys, remaining, new_partial))

    return full_solutions


# ==========================================
# Part 2: Core Logic (_solve_core)
# ==========================================

def _setup_shifted_points(points, k, shift_val):
    """Applies coordinate shift and calculates degrees."""
    shifted = [(p[0] + shift_val, p[1]) for p in points]
    m = len(shifted)
    d = (m + k - 1) // 2
    return shifted, m, d


def _compute_p_coeffs(x_vals, y_vals, k, m, c_vars):
    """Computes coefficients of P(x; c) via interpolation."""
    L_expr = sp.interpolating_poly(m, x, x_vals, y_vals)
    R_expr = sp.prod([(x - xi) for xi in x_vals])
    S_expr = sum(c_vars[j] * x ** j for j in range(k))

    p_poly = sp.Poly(sp.expand(L_expr + S_expr * R_expr), x)
    max_deg = m + k - 1
    return {i: p_poly.coeff_monomial(x ** i) for i in range(max_deg + 1)}


def _compute_convolution_numerator(range_indices, a_terms, a0):
    """Computes the sum of a_j * a_{r-j} with manual exponent handling."""
    sum_num, sum_max_exp = 0, 0
    term_exps = [a_terms[j][1] + a_terms[range_indices.stop - 1 - j + range_indices.start][1]
                 for j in range_indices]

    if term_exps: sum_max_exp = max(term_exps)

    for idx, j in enumerate(range_indices):
        r_idx = range_indices.stop - 1 - j + range_indices.start
        num = a_terms[j][0] * a_terms[r_idx][0]
        diff = sum_max_exp - term_exps[idx]
        if diff > 0: num *= (2 * a0) ** diff
        sum_num += num

    return sum_num, sum_max_exp


def _compute_a_terms_recursive(d, p_coeffs, a0):
    """Generates a_r terms where a_r = numerator / (2*a0)^exp."""
    a_terms = {0: (a0, 0)}
    for r in range(1, d + 1):
        sum_num, max_exp = _compute_convolution_numerator(range(1, r), a_terms, a0)

        # Formula: a_r = (P_r * (2a0)^max_exp - sum_num) / (2a0)^(max_exp + 1)
        P_r = p_coeffs.get(r, 0)
        new_num = P_r * (2 * a0) ** max_exp - sum_num
        a_terms[r] = (sp.expand(new_num), max_exp + 1)
    return a_terms


def _build_gb_equations(d, m, k, p_coeffs, a_terms, a0):
    """Builds the system of equations for the Gröbner basis."""
    # Initial equation: a_0^2 - P_0 = 0
    eqs = [sp.expand(a0 ** 2 - p_coeffs.get(0, 0))]

    max_deg = m + k - 1
    for r in range(d + 1, max_deg + 1):
        start_j, end_j = max(0, r - d), min(r, d)
        if start_j > end_j: continue

        q2_num, q2_exp = _compute_convolution_numerator(range(start_j, end_j + 1), a_terms, a0)
        # Eq: q^2_coeff - P_r = 0  =>  q2_num - P_r * (2a0)^exp = 0
        P_r = p_coeffs.get(r, 0)
        eqs.append(sp.expand(q2_num - P_r * (2 * a0) ** q2_exp))
    return eqs


def _reconstruct_poly_from_sol(sol, a_terms, d, shift_val, a0):
    """Reconstructs a single valid polynomial from a numeric solution."""
    #if sol[a0] == 0: return None

    print(f"Sol = {sol}")
    coeffs = {}
    for r in range(d + 1):
        num_poly, exp_val = a_terms.get(r, (0, 0))
        try:
            #val = num_poly.subs(sol) / ((2 * sol[a0]) ** exp_val)
            val = num_poly.subs(sol) / ((2 * a0) ** exp_val)
            if not (val.is_number and val.is_rational): return None
            coeffs[r] = val
        except ZeroDivisionError:
            return None

    Q_t = sum(coeffs[i] * x ** i for i in range(d + 1))
    return Q_t.subs(x, x + shift_val)


def _solve_core(points, k, shift):
    """Orchestrator for the core algebraic solving logic."""
    pts, m, d = _setup_shifted_points(points, k, shift[0])

    # Setup variables
    # shifted polynomial has a0^2 = y_i, where i is the shift value
    a0 = sp.sqrt(shift[1]) # or a0 = -sp.sqrt(shift[1])
    c_vars = [sp.Symbol(f'c_{k - i - 1}') for i in range(k)]
    gb_vars = c_vars

    # Compute P coeffs and recursive a_terms
    p_coeffs = _compute_p_coeffs([p[0] for p in pts], [p[1] for p in pts], k, m, c_vars)
    a_terms = _compute_a_terms_recursive(d, p_coeffs, a0)
    valid_polys = []
    if k > 0:
        # Build and solve system
        sys_eqs = _build_gb_equations(d, m, k, p_coeffs, a_terms, a0)
        try:
            print(sys_eqs)
            gb = sp.groebner(sys_eqs, gb_vars, order='lex', domain='QQ')
        except:
            return []

        print(f'gb = {gb}')
        if list(gb) == [1]: return []  # No solution
        solutions = solve_triangular_system(list(gb), gb_vars)
        valid_polys = [_reconstruct_poly_from_sol(s, a_terms, d, shift[0], a0) for s in solutions]
    else:
        valid_polys = [_reconstruct_poly_from_sol({}, a_terms, d, shift[0], a0)]
    return [p for p in valid_polys if p is not None]



# ==========================================
# Part 3: Phaseless Interpolation
# ==========================================


def _calculate_shift(points):
    for x, y in points:
        if y != 0: return -x, y
    return None


def _deduplicate_solutions(candidates):
    """Simplifies polynomials and filters out duplicates."""
    unique_polys = []
    seen_exprs = set()

    for p in candidates:
        simp_p = sp.simplify(p)
        if simp_p not in seen_exprs:
            unique_polys.append(simp_p)
            seen_exprs.add(simp_p)

    return unique_polys


def _log_final_results(solutions):
    """Prints the formatted final solutions to the console."""
    print(f"--- Results ---")
    for i, poly in enumerate(solutions):
        print(f"Solution {i + 1}: q(x) = {poly}")


def _solve_affine_square_roots(points, k):
    """
    Main driver: Orchestrates the search for affine square roots.
    1. Configures shift limits.
    2. searches for candidates.
    3. Deduplicates and logs results.
    """
    print(f"--- Configuration: Points={len(points)}, k={k} ---")

    shift = _calculate_shift(points)
    if shift is None:
        solutions = [0]
        print(f"No shift: it is the zero polynomial")
    else:
        solutions = _solve_core(points, k, shift=shift)
        print(f"Shift {shift}: Found {len(solutions)} solutions")

    unique_solutions = _deduplicate_solutions(solutions)
    _log_final_results(unique_solutions)

    return unique_solutions


def phaseless_interpolation(points, k):
    squared_points = [(x, y**2) for x, y in points]
    return _solve_affine_square_roots(squared_points, k)


if __name__ == "__main__":
    # Test Case 1: y = x
    print("Test Case 1: y = x")
    points1 = [(0, 0), (1, -1), (2, 2)]
    res1 = phaseless_interpolation(points1, k=0)
    print("\n")

    # Test Case 2: y = (x+1)
    print("Test Case 2: y = (x+1)")
    points2 = [(1, -2), (2, 3), (0, 1)]
    res2 = phaseless_interpolation(points2, k=0)
    print("\n")

    # Test Case 3: y = (x+1)^2
    print("Test Case 3: y = (x+1)^2, k=1, but wrong evaluation at x=-2")
    points3 = [(-2, 9), (0, 1), (1, 4), (2, 9)]
    res3 = phaseless_interpolation(points3, k=1)
    print("\n")

    # Test Case 4: Higher degree
    print("Test Case 4: y = x^5 - 6x^4 + 5x^3 + 4x^2 - 3x + 2 from 10 points")
    points4 = [(1, 3), (2, 12), (3, 79), (4, 138), (5, 87), (6, 1208), (-1, 3), (-2, 144), (-3, 817), (0, 2)]
    res4 = phaseless_interpolation(points4, k=1)

    # Test Case 5: Higher degree, higher degree of freedom
    print("Test Case 5: y = x^5 - 6x^4 + 5x^3 + 4x^2 - 3x + 2 from 9 points")
    points5 = [(1, 3), (2, 12), (3, 79), (4, 138), (5, 87), (6, 1208), (-1, 3), (-2, 144), (-3, 817)]
    res5 = phaseless_interpolation(points5, k=2)

    # Test Case 6: Higher degree, higher degree of freedom
    print("Test Case 5: y = x^5 - 6x^4 + 5x^3 + 4x^2 - 3x + 2 from 8 points")
    points6 = [(1, 3), (2, 12), (3, 79), (4, 138), (5, 87), (-1, 3), (-2, 144), (0, 2)]
    res6 = phaseless_interpolation(points6, k=3)

    # Test Case 7: Zero polynomial
    print("Test Case 7: y = 0 from 9 points and k=2")
    points7 = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (-1, 0), (-2, 0), (-3, 0)]
    res7 = phaseless_interpolation(points7, k=2)
