# Code index

*Status: M2. Updated at every milestone.*

## How the modules connect

```text
conventions ◄── everything (the only place rotations are defined)
body        ◄── correction, (later) simulator, step0, baselines
codec       ◄── (later) simulator writes frames, step0 reads them
manifest    ◄── scripts/check_env.py, data/diodem (download), scripts/verify_amass.py
data/net    ◄── data/diodem (HTTP session: OS trust store, User-Agent, MD5-verified download)
data/amass  ◄── synth (writes AMASS-format files), figures, (later) simulator
data/diodem ◄── scripts/download_diodem.py, scripts/make_figures.py, validation 11.x
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
| [mesqsim/data/net.py](../mesqsim/data/net.py) | HTTP session (OS trust store, User-Agent) and an atomic, size- and MD5-checked download |
| [mesqsim/data/diodem.py](../mesqsim/data/diodem.py) | DIODEM: Dataverse file list, download into the official loader's layout, manifest, chain adjacency, rate/norm scan, `load_trial` (official loader), six-axis relative-heading analysis |
| [mesqsim/data/amass.py](../mesqsim/data/amass.py) | AMASS: `.npz` loader (SMPL+H and SMPL-X keys), forward kinematics, pod → SMPL joint mapping, up-axis estimate, hip and knee flexion, file discovery |
| [mesqsim/synth.py](../mesqsim/synth.py) | synthetic AMASS-format motion (T-pose, stand, walk, circle) so tests run without licensed data |
| [mesqsim/figures.py](../mesqsim/figures.py), [mesqsim/plotstyle.py](../mesqsim/plotstyle.py) | stick figure (generic bone lengths) and the DIODEM figure; one plot style |
| [mesqsim/validation.py](../mesqsim/validation.py) | registry of the brief §9 checks: what, why, expected, command, and the computation |
| [scripts/check_env.py](../scripts/check_env.py) | prints versions, OS, CPU; checks imports and data checksums |
| [scripts/download_diodem.py](../scripts/download_diodem.py) | downloads and MD5-verifies DIODEM, writes `manifests/diodem.csv` |
| [scripts/verify_amass.py](../scripts/verify_amass.py) | records, extracts and indexes hand-downloaded AMASS archives and the SMPL+H model |
| [scripts/make_figures.py](../scripts/make_figures.py) | regenerates `docs/figures/*.png` |
| [scripts/diodem_inspect.py](../scripts/diodem_inspect.py) | reproduces the DIODEM frame, adjacency and gyroscope-offset facts in DATA.md |
| [manifests/](../manifests/) | committed manifests: path, size, SHA-256 of every data file (data itself never committed) |
| [scripts/validate.py](../scripts/validate.py) | runs every check and rewrites [VALIDATION.md](VALIDATION.md) |
| [scripts/check_all.sh](../scripts/check_all.sh) | the full check: ruff, pytest, validation |
| [tests/](../tests/) | `test_codec.py` (check 1), `test_conventions.py` (2), `test_correction.py` (3, 4), `test_amass.py` (10), `test_diodem.py` (11), `test_validation.py` (all registered checks); `conftest.py` skips data tests when data are absent |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | CI: ruff, the synthetic tests and quick validation on every push and pull request |

## Where to extend

- A new rotation helper goes in `conventions.py` with a test in `test_conventions.py`, never inline elsewhere.
- A new validation number: add a `@register(...)` function in `validation.py`; it appears in VALIDATION.md and is
  asserted by `tests/test_validation.py` automatically.
- A new dataset: a module in `mesqsim/data/` that downloads (or verifies a hand download) into `$MESQ_DATA_DIR/<name>/`,
  writes `manifests/<name>.csv`, converts to the +Z-up world at load time with a test of that conversion, and a
  `conftest.py` fixture that skips its tests when the data are absent.
