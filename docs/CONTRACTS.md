# Contracts And Validation Boundaries

This project follows a design-by-contract style:

- Public/boundary APIs validate and normalize inputs aggressively.
- Internal algorithm kernels assume prevalidated inputs and focus on math.
- Internal checks are kept only where they prevent silent numerical corruption.

This keeps responsibilities clear without scattering repetitive validation logic.

## Boundary Components (strict validation)

These are responsible for type/shape/range/schema checks:

- `selection.GramData`
- `selection.CrossValGramData`
- selector constructors exported from `selection`
- selector entrypoints (`fit(...)`) via selector validation utilities
- parameter validators in `selection.validation.interface_validation` and `selection.validation.index_validation`

Boundary misuse should raise clear `TypeError`/`ValueError`.

## Internal Components (assume valid inputs)

These primarily assume contract-satisfying inputs:

- `selection.core.incremental_solver`
- `selection.selectors.routines_cv_scoring`
- helper algebra in `selection.core.state_ops` and `selection.core.solvers`

Internal checks are retained only for numerical safety, e.g.:

- near-singular pivots,
- non-finite factorization outputs,
- unstable backward downdates.

## Practical Rules For Contributors

1. Add new input validation at public boundaries, not in deep kernels.
2. Document preconditions/postconditions in docstrings for internal helpers.
3. Keep internal runtime guards when removing them risks silent invalid math.
4. Prefer deterministic tie-breaking and explicit failure semantics.
5. If a contract changes, update:
   - docstrings,
   - `docs/PORTING.md`,
   - tests that assert boundary behavior.

## Failure Semantics

- Boundary misuse: `TypeError` or `ValueError`.
- Ill-conditioned linear algebra in strict paths: `np.linalg.LinAlgError`.
- Screened/invalid candidates in internal search helpers: `None` or `np.inf` scores.
