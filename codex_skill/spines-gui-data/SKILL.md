---
name: spines-gui-data
description: interpret, locate, and inspect spinesgui conversion libraries for dendrite, dendritic spine, and parent-child roi relationship tasks. use only after a prior step has already identified the relevant userID, animalID, and expID, such as the lab-data-access skill. use when working with dendrites and dendritic spines, mapping calcium-data roi indices to suite2p plane roi ids, tracing soma-dendrite-spine hierarchy, comparing apical and basal dendrites, or identifying axon-bouton relationships. do not use for bouton-only tasks that do not require parent axon relationships.
---

# Spines GUI Data

Interpret SpinesGUI conversion libraries that describe ROI identity, hierarchy, plane placement, and mapping back to Suite2p outputs. Load and inspect the `.npy` conversion file directly when the user needs facts from a specific experiment.

Assume `userID`, `animalID`, and `expID` were already resolved upstream. This skill should normally be used only after the `lab-data-access` skill or an equivalent earlier step has already identified the correct repository location.

## Core workflow

1. Use the already-resolved `userID`, `animalID`, and `expID` from the prior step. Do not guess them.
2. Check whether `/home/{userID}/data/Repository/{animalID}/{expID}/suite2p` exists.
3. Check whether `SpinesGUI` exists inside that `suite2p` directory.
4. Look for one of these files inside `SpinesGUI`:
   - `ROIs_dendrite_axon_mode_conversion.npy`
   - `ROIs_normal_mode_conversion.npy`
5. Infer the interpretation mode from the filename.
6. If the file is missing for the requested `expID`, search same-day experiments for the same animal. Some days store the conversion library only in the first processed `expID`, and same-day experiments often share ROI numbering.
7. Load the file with `scripts/inspect_conversion.py` when the user needs direct inspection, summary, or validation.
8. Explain the result using the terminology and schema rules below.

## When to use the skill output

Use this skill when the user needs to:
- identify whether an ROI is a soma, parent dendrite, dendritic spine, parent axon, or axonal bouton
- recover parent-child relationships, especially dendrite-spine and axon-bouton relationships
- compare dendrite classes such as apical vs basal, when the hierarchy needs to be understood from the conversion library
- map calcium-data ROI indices to Suite2p plane-specific ROI ids
- trace an ROI back to its plane and original Suite2p ROI entry
- confirm whether a requested experiment reuses same-day conversion data

Do not use this skill for:
- bouton-only analysis when no parent axon relationship is needed
- generic calcium-analysis tasks that do not require ROI hierarchy interpretation
- early repository-discovery steps before `userID`, `animalID`, and `expID` are already known

## File interpretation rules

Determine the mode from the filename, not from guesswork.

### Normal mode

File: `ROIs_normal_mode_conversion.npy`

Interpret `roi-type` codes as:
- `0` = cell
- `1` = parent dendrite
- `2` = dendritic spine

For the `roi-type` list in normal mode, interpret positions as:
- position 1: `roi-type`
- position 2: `soma-id`
- position 3: `dendrite-id`
- position 4: `dendritic-spine-id`

Relationship rules:
- one soma can be associated with multiple dendrite ids
- each dendrite id belongs to only one soma id
- one dendrite can be associated with multiple dendritic spine ids
- each dendritic spine id belongs to only one dendrite id
- `0` means the ROI does not belong to that category in that field

Example:
- a soma may have `soma-id = 2`, `dendrite-id = 0`, `dendritic-spine-id = 0`

#### Interpretation note for parent dendrites within the same experiment

Within the same `expID`, different parent dendrite ROIs can still be biologically related to one another. They may represent:
- different portions of the same dendrite
- different branches that arise from one common dendritic arbor

Do not assume that different parent dendrite ids always correspond to fully unrelated dendrites. Treat them as distinct annotated parent dendrite ROIs in the conversion library, while noting that some may still belong to the same larger dendritic structure.

### Dendrite/axon mode

File: `ROIs_dendrite_axon_mode_conversion.npy`

Interpret `roi-type` codes as:
- `0` = parent dendrite
- `1` = dendritic spine
- `2` = parent axon
- `3` = axonal bouton

For the `roi-type` list in dendrite/axon mode, interpret positions as:
- position 1: `roi-type`
- position 2: `dendrite-id`
- position 3: `dendritic-spine-id`
- position 4: `axon-id`
- position 5: `axonal-bouton-id`

Relationship rules:
- one dendrite can be associated with multiple dendritic spine ids
- each dendritic spine id belongs to only one dendrite id
- one axon can be associated with multiple axonal bouton ids
- each axonal bouton id belongs to only one axon id
- `0` means the ROI does not belong to that category in that field

Example:
- a parent dendrite may have `dendrite-id = 2`, `dendritic-spine-id = 0`, `axon-id = 0`, `axonal-bouton-id = 0`

## Key meanings

Each top-level dictionary key is a **general ROI ID**. These IDs may be non-contiguous because of GUI-related issues.

Do not confuse these identifiers:
- **general ROI ID**: the top-level key in the conversion library
- **plane ROI ID**: the Suite2p ROI index inside a single plane folder
- **conversion_index**: the ROI index used by the calcium data, often referenced elsewhere as a neuron index

## Multi-animal handling

Treat conversion libraries independently per animal.

Do not assume that the following identifiers are comparable across animals:
- general ROI IDs
- dendrite ids
- axon ids
- conversion indices

These identifiers are experiment-specific and may also be date-specific, but they are not globally consistent across animals.

When working with multiple animals:
- load and interpret each animal’s conversion library separately
- apply same-day fallback rules independently per animal
- only compare derived quantities such as counts, averages, or relationship patterns, not raw ids
- only build per-animal global libraries or cross-animal summaries if the task explicitly asks for them

Avoid statements that directly compare ids across animals, such as “dendrite 3 in animal A vs dendrite 3 in animal B”, because those ids do not refer to equivalent biological structures.

For each ROI entry, expect these fields when present:

### `roi-type`
A mode-dependent list that encodes the ROI class and hierarchical ids.

### `plane`
The imaging plane from which the ROI was taken. This matters for multiplane recordings and for tracing back to the original Suite2p outputs.

### `ROI coordinates`
An array describing where the ROI was placed in the original binary data used for calcium signal extraction.

### `conversion`
A two-value list:
- first value: plane
- second value: ROI id within that plane

Use this to trace the ROI back to the correct Suite2p plane folder. In multiplane Suite2p outputs, ROI numbering restarts from `0` inside each plane folder.

### `conversion_index`
The ROI index used in the calcium-data arrays.

## Same-day fallback rule

If the conversion library is missing for the requested `expID`, search other experiments from the same date for the same animal before concluding that the data is unavailable.

When using a same-day fallback:
- state the original requested `expID`
- state the fallback `expID` actually used
- say that same-day experiments often share ROI numbering and conversion data
- mention that some days store the conversion library only in the first processed experiment

## Output expectations

When answering the user, prefer this structure:
1. file path used
2. mode detected
3. whether the requested experiment or a same-day fallback was used
4. ROI identity or relationship requested
5. relevant ids: general ROI ID, hierarchy ids, plane, plane ROI ID, and `conversion_index` when available
6. any ambiguity, missing fields, or non-contiguous ids that matter for interpretation

When describing parent dendrites from the same experiment, note that distinct parent dendrite ids may still correspond to connected portions or branches of the same larger dendritic structure.

## Use the bundled script

Use `scripts/inspect_conversion.py` when the user asks to inspect a real conversion file or when you need deterministic summaries.

Run it with the resolved identifiers from the upstream step:

```bash
python scripts/inspect_conversion.py <user_id> <animal_id> <exp_id>
python scripts/inspect_conversion.py <user_id> <animal_id> <exp_id> --general-roi-id 154
python scripts/inspect_conversion.py <user_id> <animal_id> <exp_id> --conversion-index 27
```

The script can:
- locate the conversion file for a requested experiment
- fall back to same-day experiments
- detect the mode from the filename
- summarize the library
- inspect one ROI by general ROI ID or by `conversion_index`

## Example requests

- "Find the parent dendrite for ROI 154."
- "Map calcium ROI 42 back to its Suite2p plane and ROI id."
- "Is this ROI a soma, dendrite, spine, axon, or bouton?"
- "Compare apical and basal dendrite groups, and make sure the spine relationships are interpreted correctly."
- "This bouton list is confusing. Tell me which parent axon each bouton belongs to."
