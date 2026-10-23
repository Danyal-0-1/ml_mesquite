# Code index

*Status: M1. Updated at every milestone.*

## How the modules connect

```text
conventions ◄── everything (the only place rotations are defined)
body        ◄── correction, (later) simulator, step0, baselines
codec       ◄── (later) simulator writes frames, step0 reads them
manifest    ◄── scripts/check_env.py, (later) data download and verification
validation  ◄── scripts/validate.py, tests/test_validation.py
```

## Files

| file | purpose |
|---|---|
| [mesqsim/conventions.py](../mesqsim/conventions.py) | quaternion order, Hamilton product, Y(α), heading, wrap, Exp/Log, slerp, circular statistics |
| [mesqsim/codec.py](../mesqsim/codec.py) | the 16-byte wire frame: firmware quantisation, encode, decode, stream resync |
| [mesqsim/body.py](../mesqsim/body.py) | the 15 rendered pods and the 14-joint tree (parent before child) |
| [mesqsim/correction.py](../mesqsim/correction.py) | targets δ = wrap(ψ_c − ψ_p); outward correction ĉ; q̃ = Y(ĉ)·q |
| [mesqsim/manifest.py](../mesqsim/manifest.py) | data manifests (path, size, SHA-256): read, write, verify, record |
| [mesqsim/paths.py](../mesqsim/paths.py) | data and output folders (`MESQ_DATA_DIR`, `MESQ_OUTPUT_DIR`) |
| [mesqsim/validation.py](../mesqsim/validation.py) | registry of the brief §9 checks: what, why, expected, command, and the computation |
| [scripts/check_env.py](../scripts/check_env.py) | prints versions, OS, CPU; checks imports and data checksums |
| [scripts/validate.py](../scripts/validate.py) | runs every check and rewrites [VALIDATION.md](VALIDATION.md) |
| [scripts/check_all.sh](../scripts/check_all.sh) | the full check: ruff, pytest, validation |
| [tests/](../tests/) | `test_codec.py` (check 1), `test_conventions.py` (2), `test_correction.py` (3, 4), `test_validation.py` (all registered checks) |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | CI: ruff, the synthetic tests and quick validation on every push and pull request |

## Where to extend

- A new rotation helper goes in `conventions.py` with a test in `test_conventions.py`, never inline elsewhere.
- A new validation number: add a `@register(...)` function in `validation.py`; it appears in VALIDATION.md and is
  asserted by `tests/test_validation.py` automatically.
