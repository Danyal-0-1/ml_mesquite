# CLAUDE.md: rules for every session in this repository

This repository holds Step 1 of the Mesquite ML project (project owner: **Danyal Khorami**). A collaborator runs the
sessions. The full brief is [docs/brief/STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md); read it before working. Do not
edit the brief.

## Rules

1. **The brief is the specification.** Where it is silent, make a choice, write it down (WORKLOG.md and the relevant
   doc) and test it. Where only a person can act (registering for a dataset) or a decision would change the research,
   stop and ask the collaborator.
2. **Keep it private.** The repository and its results stay private until Danyal says otherwise. Before any
   `git push`, confirm the GitHub repository is private
   (`curl -s https://api.github.com/repos/Danyal-0-1/ml_mesquite` must return 404 without credentials). On 2026-10-22
   it was public, so all work was committed locally only.
3. **No data in git.** Never commit datasets, body-model files, downloaded archives, generated datasets, model outputs
   or credentials. The AMASS and SMPL+H licences forbid redistribution.
   - Data live in `data/` and outputs in `outputs/` (both git-ignored), or wherever the environment variables
     `MESQ_DATA_DIR` and `MESQ_OUTPUT_DIR` point.
   - Commit instead: download instructions and scripts, manifests (file, size, SHA-256), small synthetic fixtures.
   - Every committed file stays under 5 MB.
4. **Work in milestones** (brief §7). For each: create a branch, implement, run `scripts/check_all.sh`, update the docs
   and `WORKLOG.md`, commit, push (only once private), open a pull request into `main` with what changed and the
   check output. Merge only when the collaborator says Danyal approved. Never force-push `main`.
5. **Never invent data, results or numbers.** Every number in the docs comes from a command that was run (name it) or
   a cited source. Every simulator parameter carries a label: `datasheet`, `firmware`, `assumption` or `measured`.
6. **Tests run without licensed data.** Unit and property tests use synthetic motion. Tests that need AMASS or DIODEM
   are marked (`@pytest.mark.amass`, `@pytest.mark.diodem`) and skip cleanly when the data are absent.
7. **One convention per quantity, tested.** `mesqsim/conventions.py` owns quaternion order `(x, y, z, w)`, the
   Hamilton product, body→world orientations, +Z-up world, `Y(α)`, heading (intrinsic Z-Y-X) and `wrap` into
   (−180°, 180°]. Never re-implement these elsewhere; import them.
8. **Talk to the collaborator.** Before each milestone, say in plain words what you will do. After it, say what you
   did and how you checked it.
9. **Stop at the end of Step 1, or when blocked.** Then write the report (brief §11). Do not build the geometric
   estimator or any learned model; those are Step 2.

## Practicalities

- Python 3.11. On the collaborator's machine the virtual environment is `D:\mesquite_work\venv` and the data and
  outputs live in `D:\mesquite_work\data` and `D:\mesquite_work\outputs` (outside OneDrive).
- Run the full check with `bash scripts/check_all.sh`.
- Radians inside the code, degrees in reports; put the unit in the variable name (`_rad`, `_deg`, `_s`, `_ms`).
- `labels.parquet` is simulation truth. Step 0 and every estimator must never read it.
