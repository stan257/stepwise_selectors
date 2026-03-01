# Contracts And Validation Boundaries

This project follows a design-by-contract style:

- Public/boundary APIs validate and normalize inputs aggressively.
- Internal algorithm kernels assume prevalidated inputs and focus on math.
- Internal checks are kept only where they prevent silent numerical corruption.

This keeps responsibilities clear without scattering repetitive validation logic.

## Boundary Components (strict validation)

These are responsible for type/shape/range/schema checks:

- `selection.definitions.GramData`
- `selection.definitions.CrossValGramData`
- selector constructors in `selection.routines*` and `selection.grouped_routines`
- selector entrypoints (`fit(...)`) via `selection.selector_validation`
- parameter validators in `selection.interface_validation` and `selection.index_validation`

Boundary misuse should raise clear `TypeError`/`ValueError`.

## Internal Components (assume valid inputs)

These primarily assume contract-satisfying inputs:

- `selection.incremental_solver`
- `selection.routines_cv_scoring`
- helper algebra in `selection.state_ops` and `selection.solvers`

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
