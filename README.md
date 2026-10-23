# Does the Adaptive Shift Adapt? Degenerate Origins in Convex-Hull Normalisation

Code, configurations, logs and results for the paper

> D. Emmanuel. *Does the Adaptive Shift Adapt? Degenerate Origins in Convex-Hull Normalisation for Skeleton-Based Multi-Entity Action Recognition.*

The repository contains:

* **CHASE+** and **CHASE+B**, the corrected origin normalisation. CHASE+B adds a hard effective-support floor.
* the **fixed-origin** and **vanilla** baselines,
* the **probes** used to analyse the released and retrained CHASE models (effective support, padding, translation, permutation),
* the **exact configuration** of every run, all **training logs** and **result files**, and the scripts that recompute every table and figure.

The trained checkpoints (25 models, 144 MB) are distributed separately: `chase-degenerate-origins-checkpoints.zip`, see [Checkpoints](#checkpoints).

All code plugs into the official CHASE repository ([Necolizer/CHASE](https://github.com/Necolizer/CHASE)), pinned at commit `0a62973af96054f24f650b21c22d974c1d482387`. The CHASE training loop, feeders and CTR-GCN backbone are used unchanged.

---

## 1. Repository layout

```
install.sh                    prepares a pinned CHASE checkout with everything below
src/
  chase_plus.py               CHASE+ / CHASE+B module (mask, centred CLB, hierarchical weights,
                              Rényi-2 band, bounded logits, Sinkhorn divergence) + self-tests
  chase_plus_repo.py          adapter: CHASE+ as a CHASE wrapper (returns the auxiliary loss)
  fixed_origin.py             fixed-origin (mode=joint) and vanilla (mode=none) wrappers
patches/
  loss_LSCE_AUX_GROUP.patch   adds the loss used with CHASE+/CHASE+B to utils/loss.py
probes/
  probe_chase.py              E1: N_eff, padding weight, translation and permutation tests
  probe_chase_v2.py           E2/E4: selected point, padding control, permutation decomposition
  probe_chase_plus.py         E5/E6: the same diagnostics for CHASE+ / CHASE+B
configs/                      exact YAML of every arm (only dataset_dir was edited)
scripts/
  calibrate_eps.py            Sinkhorn epsilon calibration (frozen per run)
  run_cad_experiments.sh      trains all arms and seeds
  compute_stats.py            reproduces Tables 4 and 5 from the result CSVs
  make_figures.py             reproduces Figures 1-5
  verify_numpy.py             NumPy check of the analytical results (25 checks)
notebooks/                    the Kaggle notebooks that produced every result (01-08)
environment/                  package lists for Environment A and Environment B
results/                      every output used in the paper (CSV, JSON, logs)
```

## 2. Experiments and where each result comes from

| ID | What | Notebook | Results | Paper |
|----|------|----------|---------|-------|
| E1 | Probe the released CAD checkpoint: N_eff, padding, translation, permutation | `01` | `results/E1_released_probe_cad/` | Tab. 1-2, Fig. 2-3 |
| E2 | Controls on the released CAD checkpoint: selected point, padding control, permutation decomposition, centroid origin | `02` | `results/E2_released_probe_cad_v2/` | Sec. 7.2-7.3, Fig. 4 |
| E1/E2 | The same on NTU Mutual 11 (X-View) | `03` | `results/E1_released_probe_ntu11/`, `results/E2_released_probe_ntu11_v2/` | Tab. 1-2, Fig. 2-4 |
| E3 | Train vanilla, CHASE, fixed origin on CAD, seeds 1-3 (Env. A) | `04` | `results/E3_train_envA/` | Tab. 4-5, Fig. 5 |
| E4 | Probe the three retrained CHASE models | `05` | `results/E4_probe_trained_chase/` | Tab. 3 |
| E5 | Train CHASE+ seeds 1-5; CHASE seeds 4-5; fixed seed 4 (Env. A); probe CHASE+ | `06` | `results/E5_chaseplus_envA/` | Tab. 4-6, Fig. 2-3, 5 |
| E6 | Train CHASE+B seeds 1-5; fixed seed 5 (Env. A); probe CHASE+B | `07` | `results/E6_chaseplusB_envA/` | Tab. 4-6, Fig. 2-3, 5 |
| E7 | CHASE and fixed origin in the authors' software stack (Env. B), seed 1 | `08` | `results/E7_envB_authors_stack/` | Tab. 7 |

`results/merged/cad_all_arms_envA.csv` contains every Environment A run (best and final-epoch accuracy). `python scripts/compute_stats.py` prints Tables 4 and 5 from it.

`results/E4_probe_trained_chase/collapse_trained_ckpts.csv` contains the Table 3 values. Running notebook `05` with the checkpoints regenerates the full file.

## 3. Environments

All runs used Kaggle with **2 x NVIDIA T4** (16 GB each).

| | Environment A (E1-E6) | Environment B (E7) |
|---|---|---|
| Python | Kaggle default (3.10/3.11) | 3.8 |
| PyTorch | Kaggle default 2.x (version printed by each notebook) | 1.9.0+cu111 |
| Other | `environment/requirements-envA.txt` | `environment/requirements-envB.txt` (pins of the CHASE `requirements.txt`) |
| GPUs / batch | 2 x T4, batch 256 (128 per GPU) | 2 x T4, batch 256 |

The published CHASE configuration specifies 6 GPUs. With 2 GPUs, the per-device batch-normalisation batch is larger; this is discussed in the paper as a candidate cause of the reproduction gap.

## 4. Data

* **Collective Activity Dataset (CAD)**: the keypoint version prepared by COMPOSER (17 COCO joints per person, 6 person slots, 10 frames). Follow the *Dataset Preparation* section of [hongluzhou/composer](https://github.com/hongluzhou/composer), as the CHASE README instructs. Place the `collective` folder (containing `tracks_normalized.pkl`, `tracks_normalized_with_person_action_label.pkl` and `joints/`) at `CHASE/NeurIPS24/data/collective`, or edit `dataset_dir` in `configs/cad_*.yaml`. The CAD test feeder expects `generated_dataset_stat/collective/joints/stats_train.pickle`; it is created by a training run, and notebook `05` shows how to create a placeholder for inference-only use.
* **NTU RGB+D 60, Mutual 11 (X-View)**: raw skeletons `nturgbd_skeletons_s001_to_s017.zip` from the [ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition) (registration required). Notebook `03` extracts the 11 mutual classes (A050-A060) and runs the CTR-GCN preprocessing that CHASE uses. The resulting `NTU60_XView.npz` is renamed `NTU60_CV.npz`, the name the CHASE config expects.
* **Released CHASE checkpoints** (probed in E1/E2, not redistributed here):
  * CAD: `https://huggingface.co/Necolizer/CHASE/resolve/main/CAD_CTR-GCN_CHASE/cad_chas.pt`
  * NTU Mutual 11 X-View: `https://huggingface.co/Necolizer/CHASE/resolve/main/NTU11_XView_CTR-GCN_CHASE/ntu11_xview_chas.pt`

## 5. Reproducing the results

### 5.1 Local machine (2 GPUs)

```bash
bash install.sh ./CHASE                 # clone CHASE at the pinned commit and add our files
cd CHASE/NeurIPS24
pip install -r ../../environment/requirements-envA.txt
# place CAD at data/collective (Section 4)

python model/Wrapper/chase_plus.py      # self-tests of CHASE+ / CHASE+B (CPU, ~10 s)
bash ../../scripts/run_cad_experiments.sh   # all arms and seeds (about 40 GPU-hours on 2 x T4)
```

Single run, for example CHASE+B seed 3:

```bash
python main_group.py --config cad_chaseplusB.yaml --seed 3 --work_dir runs/cad_chaseplusB_s3
```

Probe a trained CHASE+/CHASE+B model:

```bash
python probe_chase_plus.py --config cad_chaseplusB_probe.yaml \
       --weights runs/cad_chaseplusB_s3/cad_chaseplusB_s3.pt --out probe_cad_chaseplusB_s3
```

Probe the released CHASE checkpoint (E1, E2):

```bash
wget https://huggingface.co/Necolizer/CHASE/resolve/main/CAD_CTR-GCN_CHASE/cad_chas.pt
python probe_chase.py    --config cad_chase.yaml --weights cad_chas.pt --out probe_cad
python probe_chase_v2.py --config cad_chase.yaml --weights cad_chas.pt --out probe_cad_v2
```

### 5.2 Kaggle (as in the paper)

Each notebook in `notebooks/` is self-contained: it clones CHASE, writes the files from this repository into it, builds the configs and runs. Settings: **Accelerator GPU T4 x2**, **Internet On**, then **Save Version > Save & Run All**. Attach the CAD dataset (and, for `05`-`07`, the outputs of the earlier notebooks) with **Add Input**. Kaggle sessions stop after 12 h, so the training notebooks resume from their result CSV.

### 5.3 Tables and figures from the stored results

```bash
python scripts/compute_stats.py         # Tables 4 and 5
cd scripts && python make_figures.py    # Figures 1-5 (writes PDF and PNG in the current folder)
python scripts/verify_numpy.py          # numerical check of the propositions (no GPU, no torch)
```

## 6. Configuration of every arm

All arms share the released CHASE CAD configuration (`config/CTRGCN/cad/cad_chase.yaml` at the pinned commit) except for the wrapper and the loss:

| Setting | Value |
|---|---|
| Backbone | CTR-GCN (`model.CTRGCN.ctrgcn.Model_GROUP`), 2D input, 17 joints, 6 persons, 10 frames, 4 group classes, 5 person classes |
| Optimiser | SGD, Nesterov, lr 0.1, weight decay 4e-4, warm-up 5 epochs, decay x0.1 at epochs 25 and 45, 55 epochs |
| Batch | 256 (train and test), 2 GPUs |
| Augmentation | released CAD feeder: horizontal flip, horizontal and vertical moves (with perturbation), agent dropout |
| Evaluation | every 3 epochs; **best test accuracy** (as in CHASE) and **final-epoch accuracy** are both reported |
| Seeds | 1-5 (vanilla 1-3) |

| Arm | Wrapper | Wrapper arguments | Loss |
|---|---|---|---|
| Vanilla | `FixedOriginWrapper` | `mode=none` | `LSCE_GROUP` (smoothing 0.1) |
| CHASE | `CHASWrapper` (released) | `c1=64, c2=8, pooling_seg=[1,1,1]` | `LSCE_MBMMD_GROUP` (released, weights [1, 1]) |
| Fixed origin | `FixedOriginWrapper` | `mode=joint, t=3, v=4, m=0` (the point released CHASE selects) | `LSCE_GROUP` |
| CHASE+ | `CHASEPlusRepo` | `centred, masked, hierarchical; c1=64, c2=8; kappa_lo=10, kappa_hi=255; lam_rbr=1.0; lam_sd=0.1; eps=681.95; sd_points=8; sd_pairs=4` | `LSCE_AUX_GROUP` |
| CHASE+B | `CHASEPlusRepo` | as CHASE+, plus `kappa_floor=8` (logit bound c=2.209 for n=170); `eps=665.36` | `LSCE_AUX_GROUP` |

The Sinkhorn epsilon is calibrated once per run on training data (`scripts/calibrate_eps.py`) and then frozen. The two values differ slightly because the training feeder is randomly augmented.

The checkpoint saved by `main_group.py` is the model at the **best test epoch**.

## 7. Code versions

The CHASE+ runs (E5) used an earlier revision of `chase_plus.py` / `chase_plus_repo.py` without the `kappa_floor` option. E3 used a revision of `fixed_origin.py` without the module docstring. The released versions only *add* these features: when `kappa_floor` is `None`, the computation is identical. All runs can therefore be reproduced with the files in `src/`.

The comments and self-tests in `src/chase_plus.py` use the numbering of an earlier technical report. Correspondence with the paper:

| Code | Paper |
|---|---|
| Prop. 13 (masking) | Proposition 1 |
| Prop. 10 (centred CLB) | Proposition 2 |
| Prop. 12 (hierarchical weights) | Proposition 3 |
| Prop. 9 (support band) | Proposition 4 |
| Prop. 17 (bounded logits) | Proposition 5 |
| Prop. 16 (per-sample shift invariance) | Proposition 6 |
| Prop. 15 (mean/shape split) | Proposition 7 and Corollary |

`chase_plus.py` also contains a Plücker-bone stream (`plucker_bones`). It is not used in the paper; all arms use the joint stream.

## 8. Checkpoints

`chase-degenerate-origins-checkpoints.zip` contains:

```
envA/cad_{vanilla}_s{1,2,3}.pt
envA/cad_{chase,fixed,chaseplus,chaseplusB}_s{1..5}.pt
envB/cad_{chase,fixed}_s1.pt
MANIFEST.csv     arm, seed, environment, config, best and final-epoch accuracy, SHA-256
SHA256SUMS
```

Each file is a CHASE-format `state_dict`, saved by `main_group.py` at the best test epoch. Load it with the matching config, for example:

```bash
python main_group.py --config cad_chaseplus.yaml --run_mode test --weights envA/cad_chaseplus_s5.pt
```

or pass it to the probes with `--weights`. The incomplete Environment B CHASE seed-2 run (stopped by the 12 h limit at epoch 48) is not included and not reported.

## 9. Licence and citation

The code in this repository is released under the MIT licence (`LICENSE`). CHASE, CTR-GCN, COMPOSER, CAD and NTU RGB+D keep their own licences and terms of use. Please cite the paper (`CITATION.cff`) and CHASE (Wen et al., NeurIPS 2024) if you use this code.
