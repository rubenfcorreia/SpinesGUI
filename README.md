# SpinesGUI

SpinesGUI is a PyQt5 application for viewing Suite2p data, drawing cell,
dendrite, spine, axon, and bouton ROIs, and extracting fluorescence from
registered Suite2p movies.

## Contents

- [Requirements and launch](#requirements-and-launch)
- [Expected Suite2p layout](#expected-suite2p-layout)
- [Using the GUI](#using-the-gui)
  - [Load and inspect data](#load-and-inspect-data)
  - [Create and edit ROIs](#create-and-edit-rois)
  - [ROI modes](#roi-modes)
- [ROI extraction](#roi-extraction)
- [Extraction queue](#extraction-queue)
  - [Starting the worker](#starting-the-worker)
  - [Logs](#logs)
  - [Queue Monitor](#queue-monitor)
- [Combined Suite2p data and splitting](#combined-suite2p-data-and-splitting)
  - [Restoring extracted outputs before the canonical split](#restoring-extracted-outputs-before-the-canonical-split)
- [Post-SpinesGUI helper](#post-spinesgui-helper)
- [Related tools](#related-tools)

## Requirements and launch

The project is intended to run in the `spinesGUI` conda environment. From the
project directory:

```bash
conda activate spinesGUI
python SpinesGUI.py
```

On the laboratory workstation, `SpinesGUI.bat` starts the application over SSH
and starts the extraction worker. `SpinesGUI Monitor.bat` starts the worker and
opens the queue monitor.

## Expected Suite2p layout

When loading data, select a directory containing one or more Suite2p plane
folders directly below it:

```text
<suite2p folder>/
├── plane0/
│   ├── ops.npy
│   ├── data.bin
│   └── ...
├── plane1/
└── SpinesGUI/
```

Each `planeX/ops.npy` must contain the image and registration metadata needed
by the GUI, including `meanImg`, `yrange`, and `xrange`. The GUI ignores folders
whose names contain `combined` when looking for planes.

## Using the GUI

### Load and inspect data

1. Click **Load Suite2p** and select the Suite2p directory.
2. Use the `<` and `>` buttons to move between planes.
3. Use the view buttons to select:
   - Functional Mean
   - Functional Enhanced
   - Max Projection
   - Other Channel Mean
   - Combined, with channel 1 shown in green and channel 2 in red when available
4. Adjust the dark and white percentile sliders, or click **Auto Contrast**.
5. The **Registration Viewer** shows registration information for the current
   plane.

The yellow rectangle marks the valid Suite2p image region. ROI drawing should
be performed inside that region.

### Create and edit ROIs

Click **Add ROI**, choose the ROI type and its optional parent association, then
choose one of these shapes:

- **Tracing**: click points around the ROI and finish the trace.
- **Rectangle**: drag from one corner to the opposite corner.
- **Other Polygon**: drag a bounding box for a regular polygon with 3–10 sides.

The GUI stores ROI coordinates together with the plane number and a structured
`roi-type` relationship. In Normal Mode, the hierarchy is:

```text
cell → dendrite → spine
```

In Dendrites/Axons Mode, the hierarchy is:

```text
parent dendrite → dendritic spine
parent axon → axonal bouton
```

Use **ROIs Table** to inspect the ROI records. New ROIs are confirmed in a
review dialog and saved automatically. **Clear All ROIs** removes the current
ROI set after confirmation.

### ROI modes

The default **Normal Mode** is for cell, dendrite, and spine analysis. Enable
**Dendrites/Axons Mode** only when working with dendrites or axons; this mode
uses a different ROI hierarchy and different output filenames.

ROI files are saved under the selected root folder:

```text
SpinesGUI/ROIs.npy
SpinesGUI/ROIs_dendrite_axon_mode.npy
```

When extraction is queued, the GUI also writes the worker input files:

```text
SpinesGUI/roi_data.npy
SpinesGUI/plane_data.npy
```

If an ROI file for the selected mode already exists, the GUI loads it when the
Suite2p folder is opened. It can also detect the other mode's ROI file and
switch modes when appropriate.

## ROI extraction

Click **Extract ROIs** after drawing and saving the ROIs. The GUI asks for
confirmation before saving the ROI inputs or submitting a queue job; choosing
No leaves both unchanged. The dialog includes **Force re-extraction
(overwrite existing outputs)**. It is enabled for incomplete or partial runs
and is automatically required when a previous successful extraction exists.

The worker reads `roi_data.npy` and `plane_data.npy`, uses the Suite2p
registered binaries and metadata from each plane. Before extraction, each
source `data.bin` and optional `data_chan2.bin` is copied into
`SpinesGUI/planeX/` only when the destination is missing or its file size
differs from the source. Copies use a temporary file and atomic replacement,
so interrupted or incomplete copies are retried safely.

The worker writes outputs under:

```text
SpinesGUI/planeX/
├── stat0.npy
├── stat1.npy                 # normal mode
├── stat1_dendrite_axon_mode.npy
├── stat.npy
├── F.npy
├── Fneu.npy
├── F_chan2.npy
├── Fneu_chan2.npy
├── spks.npy
└── iscell.npy
```

It also writes these files at the `SpinesGUI` root:

```text
ROIs_conversion.npy
ROIs_dendrite_axon_mode_conversion.npy
extraction_successfull.txt
```

Only the conversion file for the active mode is expected to be produced. The
success filename contains the historical spelling `successfull` and should be
checked exactly as written.

By default, a successful extraction cannot be submitted again. Choose the
re-extraction/force option when prompted to overwrite the previous extraction
outputs.

## Extraction queue

The queue is a SQLite database plus a long-running worker. The default database
is:

```text
~/code/SpinesGUI/queue/jobs.sqlite
```

### Starting the worker

The recommended startup script creates the queue directories and starts the
worker in a tmux session named `spines_queue`:

```bash
queue/start_worker_tmux.sh
```

The worker is equivalent to:

```bash
conda activate spinesGUI
python -u worker.py \
  --db ~/code/SpinesGUI/queue/jobs.sqlite \
  --poll 2
```

Starting the script while the tmux session already exists leaves the existing
worker running. The worker polls for queued jobs, claims them in FIFO order,
runs `run_extraction`, and records the result. When a new worker starts, any
jobs left in `running` by an interrupted previous worker are recovered as
`canceled` rather than remaining stuck in the monitor.

Jobs have these states:

```text
queued → running → done
                 └→ failed
queued → canceled
```

Each job records its ID, experiment ID, root folder, mode, force flag, creation
time, start/finish times, log path, and any error traceback.

### Logs

Per-job logs are written to:

```text
queue/logs/job_<job id>_<experiment id>.log
```

The worker's combined stdout/stderr stream is written to:

```text
queue/worker_stdout.log
```

### Queue Monitor

Click **Queue Monitor** in the GUI, or run it directly:

```bash
conda activate spinesGUI
python queue_monitor.py
```

The monitor refreshes automatically and shows:

- running jobs in a table with their mode, extraction/post status, start time,
  and cancellation state;
- queued jobs and their mode/force settings;
- the four most recently finished jobs;
- a live tail of the running job log or worker log;
- checkbox selection for jobs to cancel;
- one **Cancel selected** button for either queued or running jobs;
- cancellation for queued jobs and safe cancellation requests for running jobs.

Check jobs in only one table before clicking **Cancel selected**. Queued jobs
cancel immediately. A running extraction uses cooperative cancellation:
the current Suite2p stage is allowed to finish, then the worker stops at the next
safe checkpoint and records the job as canceled. Checkpoints occur between
binary-copy, configuration, ROI-statistics, fluorescence, deconvolution, and
conversion stages. Cancellation can leave partial outputs from completed
stages; rerun extraction to produce a complete output set.

## Combined Suite2p data and splitting

For a combined Suite2p run, the recommended order is:

1. Split or prepare the combined Suite2p folder so SpinesGUI can load its
   `planeX` folders.
2. Load the combined Suite2p data in SpinesGUI.
3. Draw and save the ROIs.
4. Queue and complete ROI extraction.
5. Run the combined-data split afterward.

The current `lab_pipeline` split implementation looks for:

```text
suite2p_combined/SpinesGUI/
```

and copies matching top-level artifacts into each split experiment:

- `*conversion*.npy`;
- `extraction_*.txt`;
- `*mode*.npy`.

This means conversion libraries and extraction markers can be propagated for
downstream analysis. The split does not recursively copy the complete
`SpinesGUI` folder. In particular, it does not currently propagate:

- `roi_data.npy`;
- `plane_data.npy`;
- the nested `SpinesGUI/planeX/` extraction outputs;
- unrelated files that do not match the copy patterns.

Therefore, the split output is suitable for analyses that use the propagated
conversion files, but it is not automatically a complete standalone SpinesGUI
project for reopening and continuing ROI editing in every split experiment.

### Restoring extracted outputs before the canonical split

If ROI extraction was run while the combined data was stored as
`suite2p_combined/`, the extracted files are normally under
`suite2p_combined/SpinesGUI/planeX/`. The canonical `lab_pipeline` splitter
expects `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, and `stat.npy` directly
under `suite2p/planeX/`.

Use this preparation step before running the pipeline split:

1. Make a backup of the experiment directory.
2. Set the correct `userID` and `expID` in `SpinesGUI_after_v1.py`.
3. Run the restoration helper from the repository root:

   ```bash
   conda activate lab_pipeline
   python SpinesGUI_after_v1.py
   ```

   The helper moves the existing root-level Suite2p contents into
   `suite2p/suite2p_original_files/` and copies each
   `SpinesGUI/planeX/` output into `suite2p/planeX/`.

4. Run the canonical `lab_pipeline` split and split the per-experiment outputs.

This helper is only a restoration step. Do not use an old helper that also runs
a legacy split when the canonical `lab_pipeline` split will be used.

## Post-SpinesGUI helper

`old_versions/SpinesGUI_after_v3.py` is a legacy post-processing helper. It can:

1. move the original Suite2p contents into `suite2p_original_files/`;
2. copy processed plane contents from `suite2p/SpinesGUI/planeX/` back into
   `suite2p/planeX/`;
3. patch `ops.npy` paths;
4. optionally call its older split implementation with SpinesGUI artifact
   propagation.

Its current configuration uses hard-coded `userID`, `expID`, and flags in the
script, so it should be treated as a manual/legacy utility rather than the main
pipeline entry point. Prefer the canonical `lab_pipeline` split command for
new processing. Moving this file into a `deprecated/` directory would be
reasonable once existing users have migrated, but it should not be moved until
all references and personal workflows have been updated because the script
imports local modules by name.

## Related tools

- `SpinesGUI.py`: main GUI.
- `spines_extraction.py`: headless extraction implementation.
- `queue_db.py`: SQLite queue interface.
- `worker.py`: queue worker.
- `queue_monitor.py`: queue status and log monitor.
- `registration_viewer.py`: registration inspection window.
- `old_versions/split_combined_s2p_modified_with_spinesgui.py`: legacy/local split helper.
- `old_versions/SpinesGUI_after_v3.py`: legacy post-processing helper.
