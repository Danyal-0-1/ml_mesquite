# CLAUDE.md: rules for every session in this repository

This repository holds the Mesquite ML project (project owner: **Danyal Khorami**; refer to Danyal by name, or as
they/them). A collaborator or Danyal runs the sessions. Three briefs are the specification; read all three, then
`README.md`, `WORKLOG.md` and every file in `docs/`, before working. Never edit a brief.

- [docs/brief/STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md): Step 1, the simulator, Step 0 and the baselines (done; tag
  `step1`). Its suit facts (§3), conventions (§4.1), measurement model (§4.2), data rules (§5) and environment (§6)
  still hold where the later briefs are silent.
- [docs/brief/STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md): Step 2, the geometric estimator GEO and its explanation
  pipeline. Its estimator contract (§5.2), common clock (§5.3), invariance tests (§5.9) and evaluation (§6) still hold
  where the Step 3 brief is silent.
- [docs/brief/STEP3_BRIEF.md](docs/brief/STEP3_BRIEF.md): Step 3, the Mesquite twin, its training data and the model
  kit. **This is the current scope.**

**Step 3 starts from `step2`** (option (a) of Step 3 brief §2.1, chosen on 2026-11-17: wait for Step 2). Step 2 was
finished and tagged `step2` (`fc89dbb`) on 2026-11-18; S3-M0 is on branch `s3-m0`, and S3-M1 onward follows from it.

## Rules

1. **The briefs are the specification.** Where they are silent, make a choice, write it down (`WORKLOG.md`) and test
   it. Stop and ask the person running the session when only a person can act (copying the recordings, describing
   the suit, registering for a dataset, recording), when a decision would change the research, or when a job would
   break the budget of rule 13.
2. **Visibility and disclosure.** The repository is public: Danyal removed the private notice on 2026-11-12.
   - Step 2's estimator code: answered on 2026-11-16, push to the public repository.
   - Step 3's code (the twin, the numbers fitted from Danyal's recordings, the model kit): answered on 2026-11-17,
     **keep it local.** Commit on local branches only, and push nothing of Step 3 until the person says otherwise.
3. **No data in git.** Never commit Danyal's recordings (BVH, Motive CSV and TAK, the phone app's logs) or anything
   derived from them row by row; datasets, body-model files, downloaded archives, generated datasets, trained weights,
   model outputs or credentials. The AMASS and SMPL+H licences forbid redistribution.
   - Data live in `data/` and outputs in `outputs/` (both git-ignored), or wherever `MESQ_DATA_DIR` and
     `MESQ_OUTPUT_DIR` point. The recordings are in `$MESQ_DATA_DIR/real_suit/` (`manifests/real_suit.csv`).
   - Commit instead: code, configs, download instructions, manifests (path, size, SHA-256), small synthetic fixtures
     and summary statistics. Every committed file stays under 5 MB.
   - Never write a home-folder path, password, token or cookie into a committed file; write data paths relative to
     `$MESQ_DATA_DIR` or `$MESQ_OUTPUT_DIR`.
4. **Work in milestones** (Step 3 brief §9, as in Step 2). Before each, say in plain words what you will do. For each:
   create a branch `s3-m<n>` from `main`, implement, run `bash scripts/check_all.sh`, update `WORKLOG.md`, commit,
   push only as rule 2 allows, and open a pull request into `main` with what changed and the check output. Merge only
   when the person running the session says Danyal approved. Never force-push `main`. After each, say what you did and
   how you checked it.
5. **Never invent data, results or numbers.** Every number comes from a command that was run (name it) or a cited
   source. Every parameter in `configs/default.yaml` carries a source note and a label: `datasheet`, `firmware`,
   `assumption`, `measured` or `fitted`.
   - `measured` also covers a number read directly from one of Danyal's recordings: name the file and the command.
   - `fitted` means estimated by a named command on the training or validation split or, for a twin parameter, on the
     calibration recordings (Step 3 brief §5.4); never on test data or the check recordings.
6. **Tests run without licensed data or recordings.** Unit and property tests use synthetic motion. Tests that need
   AMASS, DIODEM or the recordings are marked (`@pytest.mark.amass`, `@pytest.mark.diodem`, `@pytest.mark.real`) and
   skip cleanly when the data are absent. The Node golden test is `@pytest.mark.node`; its outputs are committed as
   fixtures so the Python side runs without Node.
7. **One convention per quantity, tested.** `mesqsim/conventions.py` owns quaternion order `(x, y, z, w)`, the
   Hamilton product, body→world orientations, +Z-up world, `Y(α)`, heading (intrinsic Z-Y-X), `wrap` into
   (−180°, 180°] and circular statistics. Never re-implement these elsewhere; import them, and add any new helper
   there, with a test. The page's three.js +Y-up world and the BVH's Euler convention are converted in one module,
   `mesqsim/twin/page_frame.py`, at the boundary only.
8. **Truth stays with the scorer.** Only the scorer and the oracles open labels, `meta.json` or the twin's truth; no
   estimator or learned model reads them, at inference or anywhere else. An estimator that uses simulator truth (GEO-K,
   the oracles) gets it through one explicit argument whose name says so, and every table marks it. Training targets
   reach training only through `mesqsim/learn/targets.py`.
9. **Each side has its own beliefs.** The truth lives in the simulator's `error_model:` and the twin's `twin:`
   sections; GEO's priors in `geometry:`; Danyal's model settings in `learning:`, left empty and documented. Nothing
   else reads the truth sections. Randomisation and sweeps change only the simulator or the twin.
10. **Causal and deployable.** An estimate at time t uses only frames that arrived at or before t (tested). Every
    model input exists in the live stream: pod frames after Step 0, and phone messages as the page receives them.
    Nothing from the truth, the future, OptiTrack or the phone app's private log enters an input.
11. **Banners.** Every result on the twin carries one line: the twin's version, "AMASS motion", and how many twin
    parameters are `fitted`, `measured`, `datasheet` and `assumption`. Every result on Danyal's recordings is
    descriptive (there are seven). Results on synthetic motion stay labelled PROVISIONAL (SYNTHETIC MOTION).
12. **`mesquite-mocap` is read-only and pinned at `87d7c3b`.** Read the clone at `$MESQ_DATA_DIR/mesquite-mocap`, never
    the owner's working copy. Record the commit in every run's `meta.json`; if a newer commit is ever used, rerun the
    page golden test and say so. Never change `mesquite-mocap`, and never touch hardware (serial port, hub, pod, phone).
13. **Disk and compute budget.** Before writing more than 20 GB under `$MESQ_OUTPUT_DIR`, or starting a job expected
    to run longer than 8 hours, state the estimate and ask. Always leave at least 15 GB free on the disk.
14. **Scope and stopping.** Build S3-M0 to S3-M9 in order, then the report (Step 3 brief §13), and stop. Do not build a
    learned model, its data loader, its loss or its training loop (Danyal writes them; short snippets of a library's
    interface are fine), the fusion of geometry with learning (Step 4), any change to `mesquite-mocap`, or anything
    that touches hardware. **Changed by the person running the session on 2026-11-18:** to save tokens, do not write
    Step 3's Markdown explanation files (`docs/step_3/`, and the hand-written doc updates of Step 3 brief §11) unless
    the person asks for one; the list to request from is in `WORKLOG.md`, entry S3-M0. Generated files
    (`docs/VALIDATION.md`, `docs/RESULTS_STEP3.md`, the dataset's own files) are still written by their commands.
    Stop earlier when blocked, and say on what.

## Practicalities

- Python 3.11. On the collaborator's machine the virtual environment is `D:\mesquite_work\venv` and the data and
  outputs live in `D:\mesquite_work\data` and `D:\mesquite_work\outputs` (outside OneDrive). On Danyal's Mac the
  environment is `.venv` (made with `uv`) and the data and outputs live next to the repository, in
  `../mesquite_work/data` and `../mesquite_work/outputs` (set `MESQ_DATA_DIR` and `MESQ_OUTPUT_DIR`).
- Another session may be working in the main folder. Never switch branches, stage or commit in a worktree that
  another session is using; work in your own `git worktree` (Step 3 so far: `../ml_mesquite_s3`).
- Stage files by explicit path (never `git add -A`): untracked local files may exist that must not be pushed.
- Tags: `step1` = end of Step 1; `step2-m<n>` and `step3-m<n>` = each merged milestone (not `s2-m<n>` or `s3-m<n>`,
  which are the branch names); `step2` and `step3` = end of each step.
- Run the full check with `bash scripts/check_all.sh`.
- Radians inside the code, degrees in reports; put the unit in the variable name (`_rad`, `_deg`, `_s`, `_ms`).
- `labels*.parquet` is simulation truth. Step 0 and every deployable estimator must never read it.
