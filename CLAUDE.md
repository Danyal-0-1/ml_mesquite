# CLAUDE.md: rules for every session in this repository

This repository holds the Mesquite ML project (project owner: **Danyal Khorami**). A collaborator or Danyal runs the
sessions. Two briefs are the specification; read both before working, and never edit either:

- [docs/brief/STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md): Step 1, the simulator, Step 0 and the baselines (done to M5).
- [docs/brief/STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md): Step 2, the geometric estimator, then the explanation
  pipeline. **This is the current work.** The Step 1 brief still holds where the Step 2 brief is silent (its facts
  about the suit, conventions, measurement model, data rules and environment).

## Rules

1. **The briefs are the specification.** Where they are silent, make a choice, write it down (WORKLOG.md and the
   relevant doc) and test it. Where only a person can act (registering for a dataset, recording the real suit) or a
   decision would change the research, stop and ask the person running the session.
2. **Visibility and disclosure.** The repository is public: Danyal removed the private notice on 2026-11-12. The
   relative-heading estimators are a possible invention-disclosure question.
   - Before the first push of Step 2 estimator code, ask once: *"Should the Step 2 estimator code be pushed to the
     public repository, or kept local?"* Follow the answer and record it in `WORKLOG.md`.
   - Until there is an answer, commit locally.
3. **No data in git.** Never commit datasets, body-model files, downloaded archives, generated datasets, model outputs
   or credentials. The AMASS and SMPL+H licences forbid redistribution.
   - Data live in `data/` and outputs in `outputs/` (both git-ignored), or wherever the environment variables
     `MESQ_DATA_DIR` and `MESQ_OUTPUT_DIR` point.
   - Commit instead: download instructions and scripts, manifests (file, size, SHA-256), small synthetic fixtures.
   - Every committed file stays under 5 MB.
   - Never write a password, token or cookie into the repository, `WORKLOG.md` or any doc.
4. **Work in milestones** (Step 2 brief §7). S2-M0 creates `main`. From then on, for each milestone: create a branch
   from `main`, implement, run `bash scripts/check_all.sh`, update the docs and `WORKLOG.md`, commit, push (rule 2),
   open a pull request into `main` with what changed and the check output. Merge only when the person running the
   session says Danyal approved. Never force-push `main`. If `gh` is missing, give the pull-request link to open by hand.
5. **Never invent data, results or numbers.** Every number in the docs comes from a command that was run (name it) or
   a cited source. Every parameter in `configs/default.yaml` carries a source note and a label: `datasheet`,
   `firmware`, `assumption`, `measured` or `fitted`. `fitted` means estimated on the training or validation split by a
   named command, never on test data.
6. **Tests run without licensed data.** Unit and property tests use synthetic motion. Tests that need AMASS or DIODEM
   are marked (`@pytest.mark.amass`, `@pytest.mark.diodem`) and skip cleanly when the data are absent.
7. **One convention per quantity, tested.** `mesqsim/conventions.py` owns quaternion order `(x, y, z, w)`, the
   Hamilton product, body→world orientations, +Z-up world, `Y(α)`, heading (intrinsic Z-Y-X), `wrap` into
   (−180°, 180°] and circular statistics. Never re-implement these elsewhere; import them. Add any new helper there,
   with a test.
8. **Estimators never read labels.** Only the scorer and the oracles open `labels_*.parquet` or `meta.json`. An
   estimator that uses simulator truth (GEO-K) gets it through one explicit argument whose name says so, and every
   table marks it as using simulator truth.
9. **The estimator has its own beliefs.** Its priors live in the `geometry:` config section, labelled like every other
   parameter. It never reads the simulator's `error_model:` section, which is the truth. Sweeps change only the
   simulator.
10. **Causal.** An estimate at time t uses only frames that arrived at or before t. This is tested.
11. **Results stay provisional until AMASS arrives.** Until AMASS is downloaded and the split is frozen, label every
    simulated result PROVISIONAL (SYNTHETIC MOTION).
12. **Talk to the person running the session.** Before each milestone, say in plain words what you will do. After it,
    say what you did and how you checked it.
13. **Scope and stopping.** Build Step 2 (S2-M0 to S2-M6), then the explanation pipeline in `docs/explain/` (S2-M7),
    then the documentation and report (S2-M8, Step 2 brief §11), and stop. Step 2 is not finished until the
    explanation pipeline exists and the full check passes. Do not build learning features, any learned model, or the
    fusion of geometry with learning; those are Steps 3 and 4. Stop earlier when blocked, and say on what.

## Practicalities

- Python 3.11. On the collaborator's machine the virtual environment is `D:\mesquite_work\venv` and the data and
  outputs live in `D:\mesquite_work\data` and `D:\mesquite_work\outputs` (outside OneDrive).
- Run the full check with `bash scripts/check_all.sh`.
- Radians inside the code, degrees in reports; put the unit in the variable name (`_rad`, `_deg`, `_s`, `_ms`).
- `labels*.parquet` is simulation truth. Step 0 and every deployable estimator must never read it.
