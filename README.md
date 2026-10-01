# RampFlow Ground Operations Lab

RampFlow is a dependency-free Python domain library for airport turnaround and ramp operations. It models operational resources and constraints rather than exposing a toy CRUD service. The package includes turnaround task scheduling, resource reservations, runway slots, stand allocation, taxi routing, fueling, deicing, baggage reconciliation, maintenance deferrals, passenger connections, crew duty, mobile resource dispatch, apron safety, NOTAM recurrence, disruption recovery, immutable operational snapshots, and per-flight ground-emissions settlement with versioned conversion rules.

The implementation uses only the Python standard library at runtime. State-changing services use explicit version, generation, reservation, or replay semantics where the real-world workflow requires them. Cross-domain workflows publish outward events only after their domain change succeeds and use operation IDs for replay safety. Public tests exercise representative invariants, import every packaged module, and the CLI provides a dependency-free health check and snapshot digest utility.

## Quick check

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m rampflow.cli health --json
```

## Package layout

- `src/rampflow/`: production domain code.
- `src/rampflow/workflows/`: cross-domain commit and event-consistency boundary.
- `tests/`: public regression and smoke tests.
- `pyproject.toml`: package metadata. Runtime dependencies are empty.

## Ground emissions settlement

`ground_emissions.GroundEmissionsLedger` settles fuel, ground-power (GPU) and
preconditioned-air (PCA) energy together per flight when the planned usage,
equipment meter export and late supplementary readings arrive on separate
sheets:

- Every reading is kept with its source (`planned`, `meter`, `supplement`).
  An explicit authority order selects the booked figure, and each energy line
  reports the variance against the plan in kWh and percent.
- Conversion rules (fuel density/calorific value, fuel/GPU/PCA CO2e factors)
  are published as immutable, numbered versions. Each settlement pins the
  version and the full factor snapshot used; publishing a new rule set never
  changes existing settlements.
- Readings may freely correct an **open** flight (the draft is rebuilt). A
  **sealed** flight rejects new readings; late evidence opens a new settlement
  version via `revise(...)`, while every sealed version and its SHA-256 digest
  remains retrievable for the audit trail.

The repository is intentionally self-contained and does not require network access or external services for its domain logic.
