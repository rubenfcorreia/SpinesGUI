# SpinesGUI conversion schema reference

## Purpose

Use this reference when the skill needs a compact reminder of how to interpret the conversion library fields.

## Identity layers

- **General ROI ID**: top-level key in the conversion dictionary. May be non-contiguous.
- **Plane ROI ID**: ROI index inside a single Suite2p plane folder.
- **Conversion index**: ROI index used by the calcium-data arrays.

## Expected location

`/home/{userID}/data/Repository/{animalID}/{expID}/suite2p/SpinesGUI/`

Expected filenames:
- `ROIs_normal_mode_conversion.npy`
- `ROIs_dendrite_axon_mode_conversion.npy`

## Normal mode

`roi-type` positions:
1. roi-type
2. soma-id
3. dendrite-id
4. dendritic-spine-id

Codes:
- 0 = cell
- 1 = parent dendrite
- 2 = dendritic spine

## Dendrite/axon mode

`roi-type` positions:
1. roi-type
2. dendrite-id
3. dendritic-spine-id
4. axon-id
5. axonal-bouton-id

Codes:
- 0 = parent dendrite
- 1 = dendritic spine
- 2 = parent axon
- 3 = axonal bouton

## Other keys

- `plane`: original imaging plane
- `ROI coordinates`: spatial placement in the binary data used for extraction
- `conversion`: `[plane, plane_roi_id]`
- `conversion_index`: calcium-data ROI index

## Same-day fallback

Some days store the conversion library only in the first processed experiment. Same-day experiments often share ROI numbering, so a same-day fallback can be valid when the requested experiment is missing the conversion file.
