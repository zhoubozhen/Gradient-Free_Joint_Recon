# fista_tranPACT

A gradient-free joint reconstruction package based on TranPACT + FISTA-TV.

## 1. Parameter Overview

The following parameters apply to both `config.json` and `cluster_config.json`.

If a parameter is mainly used only in local or cluster scenarios, it will be noted in the remarks.

------

### 1.0 `mpi` Parameter

The default value is `true`.

If non-MPI mode is needed, set `mpi` to `false` in `mpi_config.json` and run locally. In this case, the main GPU will be selected as the first GPU in `main_gpu_idxs`.

The Condor environment does **not** allow `mpi=false`.

If non-MPI mode must be used, use the following scripts instead:

```
alias init_cluster='/home/bozhen2/my_packages/fista_tranPACT/my_code/init_cluster_workdir.sh'
alias init_exp_cluster='/home/bozhen2/my_packages/fista_tranPACT/my_code/init_cluster_exp.sh'
```

That is, use other branch versions instead of this MPI version.

------

### 1.1 `binding` Parameters

| Parameter              | Type   | Meaning                                  | Current Example | Remarks                       |
| ---------------------- | ------ | ---------------------------------------- | --------------- | ----------------------------- |
| `main_gpu_idxs`        | int    | GPU indices used by the main computation | `[0,1]`         | Main GPUs for local runs      |
| `binding.prox_gpu_idx` | int    | GPU index used by the prox worker        | `2`             | Auxiliary GPU for local runs  |
| `binding`              | object | GPU selection handled by Condor          | `{}`            | Leave empty in cluster config |

Notes:

- For local runs, it is recommended to keep the main GPU and prox GPU within the same GPU group:
  - On `a9`: use `0-2` or `3-7`
  - On `ein`: use `0-3` or `4-7`
- Avoid cross-group assignment, such as:
  - On `a9`: main GPU `0`, prox GPU `3`
  - On `ein`: main GPU `1`, prox GPU `5`
- For cluster runs, GPUs are usually assigned by the scheduler, so this field is usually left as an empty object in `cluster_config.json`.

------

### 1.2 `fista` Parameters

| Parameter            | Type   | Meaning                                         | Current Example | Remarks                                                      |
| -------------------- | ------ | ----------------------------------------------- | --------------- | ------------------------------------------------------------ |
| `fista.reg`          | float  | TV regularization weight                        | `0.0001`        | Larger value means stronger regularization and smoother results. Commonly tuned. |
| `fista.lip`          | float  | Lipschitz constant / step-size-related value    | `5.0`           | One of the key parameters for FISTA stability. Commonly tuned. |
| `fista.iter`         | int    | Maximum number of FISTA iterations              | `20`            | Number of inner iterations for each outer call.              |
| `fista.prox_mode`    | int    | Prox execution mode                             | `2`             | `1`: main single-GPU mode; `2`: dual-GPU mode with main + prox. |
| `fista.prox_impl`    | string | Prox implementation type                        | `"mix"`         | `mix`: CPU+GPU mixed implementation, recommended; `cupy`: GPU implementation, may cause errors. |
| `fista.prox_iter`    | int    | Number of internal prox iterations              | `50`            | Number of iterations for the TV proximal subproblem. Usually no need to change. |
| `fista.grad_min`     | float  | Gradient stopping threshold                     | `1e-05`         | Can be used for early stopping. Usually no need to change.   |
| `fista.cost_min`     | float  | Cost-function stopping threshold                | `0.001`         | Can be used for early stopping. Usually no need to change.   |
| `fista.save_freq`    | int    | Intermediate result saving frequency            | `1`             | Save once every N iterations. Can be changed.                |
| `fista.use_check`    | bool   | Whether to enable convergence/divergence checks | `true`          | When enabled, the threshold rules below are used. Usually recommended. |
| `fista.check_iter`   | int    | Check frequency                                 | `1`             | Check once every N iterations.                               |
| `fista.rel_thr`      | float  | Relative-change convergence threshold           | `0.01`          | If the relative change remains small for consecutive checks, convergence can be declared. Commonly tuned. |
| `fista.rel_patience` | int    | Patience for convergence detection              | `2`             | Number of consecutive satisfied checks required before stopping. Can be changed. |
| `fista.rel_warmup`   | int    | Warmup iterations before convergence checking   | `2`             | No convergence check during the first few iterations. Usually unchanged. |
| `fista.div_rel_thr`  | float  | Relative-change divergence threshold            | `0.01`          | Values above this threshold can be treated as abnormal growth. Can be changed. |
| `fista.div_patience` | int    | Patience for divergence detection               | `2`             | Number of consecutive abnormal checks required before declaring divergence. Can be changed. |
| `fista.div_warmup`   | int    | Warmup iterations before divergence checking    | `2`             | No divergence check during the first few iterations. Usually unchanged. |

Notes:

- This is currently the most commonly tuned group of parameters.

------

### 1.3 `fista.runtime` Parameters

| Parameter       | Type   | Meaning                    | Current Example                                              | Remarks                                                      |
| --------------- | ------ | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `worker_script` | string | Path to prox worker script | `"/home/bozhen2/my_packages/mpi_fista_tranPACT/my_code/run_prox_worker.py"` | May be rewritten into different forms after local / cluster initialization. Not recommended to change manually. |

Notes:

- This parameter specifies the launch script for the prox worker.
- In the template, it is usually written as a placeholder.
- After initializing a workdir:
  - In local scenarios, it may be rewritten as `my_code/run_prox_worker.py`
  - In cluster scenarios, it may be rewritten as an absolute path inside the package
- This is a runtime dependency. Do not rename it manually unless necessary.

------

### 1.4 Top-Level Main Workflow Parameters

| Parameter   | Type  | Meaning                                           | Current Example | Remarks                                                      |
| ----------- | ----- | ------------------------------------------------- | --------------- | ------------------------------------------------------------ |
| `maxfun`    | int   | Maximum number of outer function evaluations      | `60`            | Controls the outer GFJR search/evaluation budget. Can be changed. |
| `stride`    | float | Spatial downsampling stride                       | `1.0`           | `1.0` means no downsampling. Larger values reduce computation. For example, `5.0` means 5x downsampling. |
| `start`     | float | Initial parameter or initial model scaling factor | `1.05`          | Used for outer initialization. Commonly tuned.               |
| `recon_opt` | int   | Reconstruction option                             | `0`             | `0`: `"homo1layer"`; `1`: `"3layer"`; `2`: `"aubry"`. Commonly changed. |
| `skullp0`   | int   | Whether to enable skull-p0-related logic          | `0`             | `0`: set p0 to zero in skull region; `1`: do not use `skull_roi`. |

Notes:

- This group is located after `fista` and controls the main workflow.

------

### 1.5 `paths` Parameters

The `paths` section defines all input data paths and the output directory used by the current experiment.

| Parameter                      | Type   | Meaning                                           | Notes                                                        |
| ------------------------------ | ------ | ------------------------------------------------- | ------------------------------------------------------------ |
| `paths.saving_dir`             | string | Output directory for the current experiment       | All reconstruction outputs, logs, intermediate FISTA results, and `args_record.json` are saved here. This is the main path to change for a new run. |
| `paths.rec_pos_path`           | string | Receiver position file                            | Binary `.DAT` file loaded as receiver coordinates. Usually shared across cases. |
| `paths.pressure_data_path`     | string | Measured / simulated pressure data file           | HDF5 input file used as the forward pressure data. Usually changes when switching cases. |
| `paths.pressure_data_key`      | string | Dataset key inside the pressure HDF5 file         | Usually `"forward"`. Do not change unless the HDF5 internal dataset name changes. |
| `paths.noise_data_path`        | string | Noise data file                                   | HDF5 noise file used when `noise != 0`.                      |
| `paths.noise_data_key`         | string | Dataset key inside the noise HDF5 file            | Usually `"forward"`. Do not change unless the HDF5 internal dataset name changes. |
| `paths.fn_path_recon0`         | string | Medium parameter table for `recon_opt = 0`        | Used when running the homogeneous 1-layer / recon0 setting.  |
| `paths.fn_path_recon1`         | string | Medium parameter table for `recon_opt = 1`        | Used when running the 3-layer / recon1 setting.              |
| `paths.fn_path_recon2`         | string | Medium parameter table for `recon_opt = 2`        | Used when running the Aubry / water-bone setting.            |
| `paths.opt_roi_path`           | string | Optical ROI file                                  | Binary ROI mask loaded and reshaped into the reconstruction volume. |
| `paths.cor_roi_path`           | string | Cortex ROI file                                   | NumPy `.npy` ROI file. Used when `onlycor = true`.           |
| `paths.skull_roi_path`         | string | Skull ROI file                                    | NumPy `.npy` skull mask. Used when `skullp0 = 0` to zero out the skull region in `opt_roi`. |
| `paths.medium_mat_path_recon0` | string | Medium geometry `.mat` file for `recon_opt = 0`   | Loaded only when `recon_opt = 0`.                            |
| `paths.medium_mat_key_recon0`  | string | Variable key inside the recon0 medium `.mat` file | Usually `"annhp"`.                                           |
| `paths.medium_mat_path_recon1` | string | Medium geometry `.mat` file for `recon_opt = 1`   | Loaded only when `recon_opt = 1`.                            |
| `paths.medium_mat_key_recon1`  | string | Variable key inside the recon1 medium `.mat` file | Usually `"annhp"`.                                           |
| `paths.ctdata_mat_path`        | string | CT data `.mat` file for `recon_opt = 2`           | Loaded only when `recon_opt = 2`.                            |
| `paths.ctdata_mat_key`         | string | Variable key inside the CT `.mat` file            | Usually `"ct_data"`.                                         |

Notes:

- `paths.saving_dir` is the only output directory field currently used by `main.py`.
- `paths.saving_root` is no longer needed if the code uses `paths.saving_dir` directly.
- The `pressure` field is also no longer needed for constructing the output path if `saving_dir` is explicitly provided.
- When starting a new experiment, usually update:
  - `paths.saving_dir`
  - `paths.pressure_data_path`
  - the corresponding `fn_path_recon*`
  - the corresponding medium / CT path if `recon_opt` changes
- Do not modify the key fields such as `pressure_data_key`, `noise_data_key`, `medium_mat_key_recon0`, `medium_mat_key_recon1`, and `ctdata_mat_key` unless the internal variable names inside the input files are different.

------

### 1.6 Other Less Frequently Changed Top-Level Parameters

| Parameter     | Type  | Meaning                                      | Current Example | Remarks                                         |
| ------------- | ----- | -------------------------------------------- | --------------- | ----------------------------------------------- |
| `onlycor`     | bool  | Whether to run only correction-related steps | `false`         | Use only cor. Disabled by default.              |
| `noise`       | int   | Noise-level control parameter                | `5`             | Used together with `noise_scale`.               |
| `noise_scale` | float | Noise amplitude scaling                      | `0.00026`       | Actual noise amplitude = `noise * noise_scale`. |

Notes:

- This group is located after `paths`.
- It is relatively less frequently changed and is usually adjusted only for specific experiments or debugging.

------

### 1.7 `runtime` Parameters

| Parameter                      | Type | Meaning                                          | Current Example | Remarks                                                      |
| ------------------------------ | ---- | ------------------------------------------------ | --------------- | ------------------------------------------------------------ |
| `runtime.debug_small`          | bool | Whether to enable small-scale debug mode         | `true`          | Often used for quick test runs. When `stride=1`, it becomes false. |
| `runtime.debug_only_one_fista` | bool | Whether to run only one FISTA call for debugging | `false`         | Useful for quick diagnosis.                                  |
| `runtime.debug_nt`             | int  | Number of time steps used in debug mode          | `4800`          | Reduces the time dimension size.                             |
| `runtime.trace_full`           | bool | Whether to enable full trace/logging             | `true`          | Useful for debugging.                                        |
| `runtime.heartbeat_sec`        | int  | Heartbeat log interval in seconds                | `60`            | Prevents long periods without output.                        |
| `runtime.log_gpu`              | bool | Whether to record GPU status                     | `true`          | Useful for checking GPU binding and usage.                   |

Notes:

- This group mainly controls runtime debugging, logging, and observability.
- It is not commonly changed.

------

### 1.8 `physics` Parameters

| Parameter             | Type        | Meaning                                                 | Current Example | Remarks                                          |
| --------------------- | ----------- | ------------------------------------------------------- | --------------- | ------------------------------------------------ |
| `physics.cb`          | float       | Background speed of sound or related physical parameter | `1.5`           | Unit is defined by the code.                     |
| `physics.f0`          | float       | Center frequency                                        | `1.0`           | Used in the wavefield/signal model.              |
| `physics.ppw`         | int         | Points per wavelength                                   | `4`             | Affects discretization accuracy and stability.   |
| `physics.fs`          | int / float | Sampling frequency                                      | `30`            | Time sampling setting.                           |
| `physics.space_order` | int         | Devito spatial discretization order                     | `10`            | Higher values usually require more computation.  |
| `physics.to`          | int / float | Time order                                              | `2`             | Specific meaning is defined by the main program. |
| `physics.nbl`         | int         | Absorbing boundary layer thickness                      | `16`            | Common Devito/PML-related parameter.             |

Notes:

- This group controls the physical model and numerical discretization.
- It is not commonly changed.

------

### 1.9 Other Parameters at the End

| Parameter          | Type   | Meaning                | Current Example | Remarks                                        |
| ------------------ | ------ | ---------------------- | --------------- | ---------------------------------------------- |
| `devito_log_level` | string | Devito log level       | `"INFO"`        | Common values include `INFO` and `WARNING`.    |
| `out_print`        | int    | Output verbosity level | `3`             | Larger values usually mean more detailed logs. |

Notes:

- These two parameters are currently located at the end of the JSON file.
- They generally control output and logging, and are not the main tuning parameters.

------

## 2. Package Structure

```
fista_tranPACT/
├── README.md
├── my_code/
│   ├── main.py
│   ├── config.json
│   ├── cluster_config.json
│   ├── run_local.sh
│   ├── run_prox_worker.py
│   ├── init_local_workdir.sh
│   ├── init_cluster_workdir.sh
│   ├── cluster_run.sh
│   └── cluster.sub
└── src/
    ├── tranPACT/
    ├── fista_tv_3d_python/
    └── gfjr_utils.py
```

Design logic:

- `src/`: core utility library and the single source of truth.
- `my_code/`: runtime entry points, template scripts, and configuration files.
- Local/cluster workdirs only copy `my_code/*`.
- `src/` is not copied into every workdir. Instead, every workdir directly references the package-level version.

------

## 3. Usage

### 3.1 Local Run

First, initialize a new workdir:

```
mkdir -p /path/to/your_workdir
cd /path/to/your_workdir
init_local
```

Then run:

```
cd my_code
bash run_local.sh
```

Local logs are written by default to:

```
../logs/YYYYMMDD_HHMM.log
```

------

### 3.2 Cluster Run

First, initialize a new cluster workdir:

```
mkdir -p /path/to/your_cluster_workdir
cd /path/to/your_cluster_workdir
init_cluster
```

Submit the job:

```
cd my_code
condor_submit cluster.sub
```

Cluster logs are usually written to:

```
../logs/
../logs/condor/
```

------

## 4. Overall Logic

This package is divided into three layers:

### 4.1 Utility Layer

Located at:

```
src/
```

Includes:

- `tranPACT`
- `fista_tv_3d_python`
- `gfjr_utils.py`

This part is the shared utility library used by all workdirs.

If the core implementations of TranPACT / FISTA are already stable, this layer usually does not need frequent changes.

------

### 4.2 Template Layer

Located at:

```
my_code/
```

Includes:

- `main.py`
- `config.json`
- `cluster_config.json`
- initialization scripts
- run scripts
- worker scripts

This layer determines what newly initialized workdirs will look like by default.

------

### 4.3 Workdir Layer

Each time an initialization script is executed, a copy of `my_code/*` is generated under the current workdir.

This means:

- Modifying `workdir/my_code/main.py` only affects the current experiment directory.
- Modifying `~/my_packages/fista_tranPACT/my_code/main.py` affects future newly initialized directories.
- Modifying `~/my_packages/fista_tranPACT/src/*` affects all workdirs.

------

## 5. Recommended Maintenance Strategy

### 5.1 Keep the Base Utilities Stable

If `tranPACT`, `FISTA-TV`, and `gfjr_utils` are already stable, avoid modifying the following directory unless necessary:

```
~/my_packages/fista_tranPACT/src/
```

Once this part is changed, all workdirs will be affected.

------

### 5.2 Modify `main.py` Inside the Workdir for Experiment Logic

If the change is only for the current experiment, such as adjusting logic, adding logs, or changing the tuning flow, prioritize modifying:

```
your_workdir/my_code/main.py
your_workdir/my_code/config.json
```

This avoids affecting other directories and avoids polluting the template.

------

### 5.3 Modify Package-Level `my_code` for Template Updates

If a change should be inherited by all newly initialized workdirs in the future, modify:

```
~/my_packages/fista_tranPACT/my_code/
```

Then re-initialize a new workdir.

------

## 6. `config.json` and `cluster_config.json`

The structures of these two configuration files are currently mostly the same.

Common differences:

- `config.json`: used for local runs and can specify fixed GPUs.
- `cluster_config.json`: usually leaves GPU binding empty and lets the cluster environment decide GPU allocation.

------

## 7. Practical Recommendations for `config.json` and `cluster_config.json`

### Local Run Recommendations

Prioritize modifying:

- `binding.main_gpu_idx`
- `binding.prox_gpu_idx`
- `paths.*`
- `runtime.debug_*`
- `fista.*`

Suitable for quick debugging, logic verification, and single-machine testing.

------

### Cluster Run Recommendations

Prioritize modifying:

- `paths.*`
- `runtime.*`
- `physics.*`
- `fista.*`

Usually, it is not recommended to hard-code GPU indices in `cluster_config.json`, unless the cluster environment uses fixed GPU positions.

------

## 8. Common Workflow Recommendations

### 8.1 New Experiment

1. Create a new workdir.
2. Initialize it using `init_local_workdir.sh` or `init_cluster_workdir.sh`.
3. Modify `my_code/config.json` inside that workdir.
4. Modify `my_code/main.py` inside that workdir if necessary.
5. Run the experiment.

------

### 8.2 Modify Only the Current Experiment

Only modify:

```
workdir/my_code/main.py
workdir/my_code/config.json
```

Do not modify the package-level template.

------

### 8.3 Update the Future Default Template

Modify:

```
~/my_packages/fista_tranPACT/my_code/*
```

Then all newly initialized workdirs will inherit these changes.

------

### 8.4 Modify Base Algorithms / Utilities

Modify:

```
~/my_packages/fista_tranPACT/src/*
```

This affects all workdirs and should be done carefully.

------

## 9. Current Version Design Principles

The current version is stabilized around the following principles:

- `src/` is the only utility library.
- `my_code/` contains templates and entry points.
- Local/cluster workdirs only copy `my_code/*`.
- Each workdir can modify its own `main.py` independently.
- Base utilities and experiment logic are maintained in separate layers.

The goal of this structure is:

1. Maintain tools centrally without copying or drifting.
2. Allow each experiment directory to modify its logic independently without interfering with others.
3. Quickly initialize both local and cluster runs from the same template.
4. Keep debugging and long-term maintenance as clear as possible.