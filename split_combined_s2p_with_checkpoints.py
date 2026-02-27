import organise_paths
import os
import glob
import numpy as np
import shutil
import grp

# ---------------------------------------------------------------------
# SpinesGUI artifact handling
# ---------------------------------------------------------------------

def _collect_spinesgui_files(spinesgui_src_dir: str):
    """Return a sorted list of absolute file paths to copy from a combined SpinesGUI folder."""
    if not os.path.isdir(spinesgui_src_dir):
        return []

    # Explicit common filenames + pattern-based fallback (covers your typical naming)
    explicit = [
        "ROIs_dendrite_axon_mode.npy",
        "ROIs_dendrite_axon_mode_conversion.npy",
        "mode.npy",
        "extraction_log.txt",
        "extraction_successfull.txt",
    ]

    patterns = [
        "*_conversion.npy",
        "*_mode.npy",
        "extraction_*.txt",
        "data_*_copy_success*.txt",
        "*conversion*.*",
        "*extraction*log*.*",
    ]

    found = set()
    for name in explicit:
        p = os.path.join(spinesgui_src_dir, name)
        if os.path.isfile(p):
            found.add(os.path.abspath(p))

    for pat in patterns:
        for p in glob.glob(os.path.join(spinesgui_src_dir, pat)):
            if os.path.isfile(p):
                found.add(os.path.abspath(p))

    return sorted(found)


def _copy_spinesgui_artifacts_into_split_exps(
    userID: str,
    expIDs: dict,
    exp_dir_processed_channel: str,
    suite2p_combined_path: str,
):
    """
    If suite2p_combined/SpinesGUI exists, copy SpinesGUI artifacts
    into each split experiment at:
        <exp_dir_processed>/suite2p/SpinesGUI/
    Returns a dict: {expID: [(src, dst), ...]} for verification.
    """
    spinesgui_src_dir = os.path.join(suite2p_combined_path, "SpinesGUI")
    files_to_copy = _collect_spinesgui_files(spinesgui_src_dir)
    if not files_to_copy:
        return {}

    print(f"Found SpinesGUI artifacts in combined folder ({len(files_to_copy)} files). Copying into split expIDs...")

    copied = {}
    for iExp in range(len(expIDs)):
        expID = expIDs[iExp]
        _, _, _, exp_dir_processed2, _ = organise_paths.find_paths(userID, expID)

        # If we're currently splitting the ch2 tree, mirror into exp_dir_processed2/ch2
        if os.path.basename(exp_dir_processed_channel) == "ch2":
            exp_dir_processed2 = os.path.join(exp_dir_processed2, "ch2")

        dest_dir = os.path.join(exp_dir_processed2, "suite2p", "SpinesGUI")
        os.makedirs(dest_dir, exist_ok=True)

        copied[expID] = []
        for src in files_to_copy:
            dst = os.path.join(dest_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            copied[expID].append((src, dst))

    return copied


def _assert_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {what}: {path}")


def _verify_spinesgui_copies(copied_map: dict):
    """Verify that each (src,dst) exists and has the same byte size."""
    for expID, pairs in copied_map.items():
        for src, dst in pairs:
            _assert_exists(dst, f"SpinesGUI artifact for {expID}")
            if os.path.getsize(src) != os.path.getsize(dst):
                raise IOError(
                    f"SpinesGUI artifact size mismatch for {expID}:\n"
                    f"  src={src} ({os.path.getsize(src)} bytes)\n"
                    f"  dst={dst} ({os.path.getsize(dst)} bytes)"
                )


def _verify_plane_outputs(exp_dir_processed2: str, plane_name: str):
    """
    Check that the essential Suite2p plane files exist and are internally consistent:
      - ops.npy exists and points to itself/reg_file
      - F/Fneu/spks second dim equals ops['nframes']
      - data.bin size matches nframes * Ly * Lx * 2 bytes (int16)
      - iscell/stat present
    """
    plane_dir = os.path.join(exp_dir_processed2, "suite2p", plane_name)
    required = ["ops.npy", "F.npy", "Fneu.npy", "spks.npy", "iscell.npy", "stat.npy", "data.bin"]
    for fn in required:
        _assert_exists(os.path.join(plane_dir, fn), f"{plane_name}/{fn}")

    ops = np.load(os.path.join(plane_dir, "ops.npy"), allow_pickle=True).item()

    # Basic path sanity
    if os.path.abspath(ops.get("ops_path", "")) != os.path.abspath(os.path.join(plane_dir, "ops.npy")):
        raise IOError(f"ops_path incorrect in {plane_dir}/ops.npy")

    if os.path.abspath(ops.get("reg_file", "")) != os.path.abspath(os.path.join(plane_dir, "data.bin")):
        raise IOError(f"reg_file incorrect in {plane_dir}/ops.npy")

    # Consistency checks
    nframes = int(ops.get("nframes", -1))
    if nframes <= 0:
        raise IOError(f"ops['nframes'] invalid ({nframes}) in {plane_dir}/ops.npy")

    F = np.load(os.path.join(plane_dir, "F.npy"), mmap_mode="r")
    Fneu = np.load(os.path.join(plane_dir, "Fneu.npy"), mmap_mode="r")
    spks = np.load(os.path.join(plane_dir, "spks.npy"), mmap_mode="r")

    for name, arr in (("F", F), ("Fneu", Fneu), ("spks", spks)):
        if arr.ndim != 2:
            raise IOError(f"{name}.npy is not 2D in {plane_dir}")
        if arr.shape[1] != nframes:
            raise IOError(f"{name}.npy frames mismatch in {plane_dir}: {arr.shape[1]} != ops['nframes'] ({nframes})")

    Ly = int(ops.get("Ly", ops.get("Lyc", 0) or 0))
    Lx = int(ops.get("Lx", ops.get("Lxc", 0) or 0))
    if Ly <= 0 or Lx <= 0:
        # fall back to meanImg shape if present
        meanImg = ops.get("meanImg", None)
        if isinstance(meanImg, np.ndarray) and meanImg.ndim == 2:
            Ly, Lx = meanImg.shape

    if Ly <= 0 or Lx <= 0:
        raise IOError(f"Could not determine frame size (Ly/Lx) from ops.npy in {plane_dir}")

    expected_bytes = nframes * Ly * Lx * 2
    actual_bytes = os.path.getsize(os.path.join(plane_dir, "data.bin"))
    if actual_bytes != expected_bytes:
        raise IOError(
            f"data.bin size mismatch in {plane_dir}:\n"
            f"  expected {expected_bytes} bytes (nframes={nframes}, Ly={Ly}, Lx={Lx})\n"
            f"  got      {actual_bytes} bytes"
        )


def split_s2p_vid(path_to_source_bin, path_to_dest_bin, frameSize, frames_to_copy, total_frames):
    frames_to_copy = np.array(frames_to_copy, dtype=float)
    blockSize = 1000

    finfo = os.stat(path_to_source_bin)
    fsize = finfo.st_size
    frameCountCalculation = fsize / (frameSize[0] * frameSize[1] * 2)

    total_frames_to_write = len(frames_to_copy)

    with open(path_to_source_bin, 'rb') as fid, open(path_to_dest_bin, 'wb') as fid2:
        # jump forward in file to start of current experiment
        start_bytes = int(2 * frameSize[0] * frameSize[1] * frames_to_copy[0])
        fid.seek(start_bytes)

        for iStart in range(1, total_frames_to_write + 1, blockSize):
            lastFrame = iStart + blockSize - 1
            lastFrame = min(lastFrame, total_frames_to_write)
            framesToRead = lastFrame - iStart + 1
            print(f'Frame {iStart + frames_to_copy[0] - 1}-{lastFrame + frames_to_copy[0] - 1} of {frameCountCalculation}')

            # read block of frames
            read_data = np.fromfile(fid, dtype=np.int16, count=frameSize[0]*frameSize[1]*framesToRead)

            # write to other file
            read_data.tofile(fid2)

    return None


def patch_all_ops_paths(userID, expID):
    """
    Recursively find every ops.npy under suite2p_combined/ and:
      - reset ops['ops_path'] to its own file path
      - rebase reg_file and reg_file_chan2 if they exist
      - save back out
    """
    _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, expID)
    suite2p_folder = os.path.join(exp_dir_processed, 'suite2p_combined')

    pattern = os.path.join(suite2p_folder, '**', 'ops.npy')
    ops_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(ops_files)} ops.npy files under {suite2p_folder!r}")

    for ops_path in ops_files:
        folder = os.path.dirname(ops_path)
        ops = np.load(ops_path, allow_pickle=True).item()
        ops['ops_path'] = ops_path
        for key in ('reg_file', 'reg_file_chan2'):
            if key in ops:
                old = ops[key]
                new = os.path.join(folder, os.path.basename(old))
                ops[key] = new
        np.save(ops_path, ops)

    print("All ops.npy files have been updated.")


def split_combined_suite2p_v3(
    userID,
    expID,
    *,
    copy_spinesgui_artifacts=True,
    delete_suite2p_combined=False,
    verify_before_delete=True,
):
    animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)

    # check if two channels have been extracted
    if os.path.exists(os.path.join(exp_dir_processed, 'ch2')):
        dataPath = [os.path.join(exp_dir_processed), os.path.join(exp_dir_processed, 'ch2')]
    else:
        dataPath = [os.path.join(exp_dir_processed)]

    # iterate over channels
    for exp_dir_processed_channel in dataPath:

        suite2p_path = os.path.join(exp_dir_processed_channel, 'suite2p')
        suite2p_combined_path = os.path.join(exp_dir_processed_channel, 'suite2p_combined')

        if not os.path.exists(suite2p_combined_path):
            os.rename(suite2p_path, suite2p_combined_path)
            print('Patching ops file from suite2p_combined folder...')
            patch_all_ops_paths(userID, expID)
            print('Done patching ops file.')

        planes_list = sorted(glob.glob(os.path.join(suite2p_combined_path, '*plane*')))

        combined_ops = np.load(os.path.join(suite2p_combined_path, 'plane0', 'ops.npy'), allow_pickle=True).item()

        expIDs = {}
        for iExp in range(len(combined_ops['data_path'])):
            expIDs[iExp] = os.path.basename(combined_ops['data_path'][iExp])

        all_animal_ids = [expIDs[iExp][14:] for iExp in range(len(expIDs))]
        if len(set(all_animal_ids)) > 1:
            raise Exception('Combined multiple animals not permitted')

        # SpinesGUI artifacts copy + record for verification
        copied_map = {}
        if copy_spinesgui_artifacts:
            copied_map = _copy_spinesgui_artifacts_into_split_exps(
                userID, expIDs, exp_dir_processed_channel, suite2p_combined_path
            )

        for iPlane in range(len(planes_list)):
            print('Plane ' + str(iPlane))
            plane_name = 'plane' + str(iPlane)

            # load the combined data
            F = np.load(os.path.join(suite2p_combined_path, plane_name, 'F.npy'))
            Fneu = np.load(os.path.join(suite2p_combined_path, plane_name, 'Fneu.npy'))
            spks = np.load(os.path.join(suite2p_combined_path, plane_name, 'spks.npy'))

            for iExp in range(len(expIDs)):
                expID2 = expIDs[iExp]
                frames_in_exp = int(combined_ops['frames_per_folder'][iExp])

                exp_start_frame = int(np.sum(combined_ops['frames_per_folder'][0:iExp]))
                exp_end_frame = exp_start_frame + frames_in_exp  # python slicing end is exclusive

                F_exp = F[:, exp_start_frame:exp_end_frame]
                Fneu_exp = Fneu[:, exp_start_frame:exp_end_frame]
                spks_exp = spks[:, exp_start_frame:exp_end_frame]

                animalID2, remote_repository_root2, processed_root2, exp_dir_processed2, exp_dir_raw2 = organise_paths.find_paths(userID, expID2)

                if os.path.basename(exp_dir_processed_channel) == 'ch2':
                    exp_dir_processed2 = os.path.join(exp_dir_processed2, 'ch2')

                out_plane_dir = os.path.join(exp_dir_processed2, 'suite2p', plane_name)
                os.makedirs(out_plane_dir, exist_ok=True)

                print('Cropping and saving cell traces...')
                np.save(os.path.join(out_plane_dir, 'F.npy'), F_exp)
                np.save(os.path.join(out_plane_dir, 'Fneu.npy'), Fneu_exp)
                np.save(os.path.join(out_plane_dir, 'spks.npy'), spks_exp)

                shutil.copy(os.path.join(suite2p_combined_path, plane_name, 'iscell.npy'), os.path.join(out_plane_dir, 'iscell.npy'))
                shutil.copy(os.path.join(suite2p_combined_path, plane_name, 'stat.npy'), os.path.join(out_plane_dir, 'stat.npy'))
                shutil.copy(os.path.join(suite2p_combined_path, plane_name, 'ops.npy'), os.path.join(out_plane_dir, 'ops.npy'))

                print('Updating ops file...')
                ops = np.load(os.path.join(out_plane_dir, 'ops.npy'), allow_pickle=True).item()
                ops['ops_path'] = os.path.join(out_plane_dir, 'ops.npy')
                ops['reg_file'] = os.path.join(out_plane_dir, 'data.bin')
                ops['frames_per_folder'] = [frames_in_exp]
                ops.pop('reg_file_chan2', None)

                # Filter filelist/frames_per_file down to this expID
                files = ops.get('filelist', [])
                frames_per_file = ops.get('frames_per_file', [])
                if files and frames_per_file and len(files) == len(frames_per_file):
                    keep_mask = [expID2 in f for f in files]
                    ops['filelist'] = [f for f, keep in zip(files, keep_mask) if keep]
                    ops['frames_per_file'] = [n for n, keep in zip(frames_per_file, keep_mask) if keep]

                # Update nframes to match what we actually saved
                ops['nframes'] = int(F_exp.shape[1])

                # Slice motion offsets to experiment frames if present
                combined_nframes = int(F.shape[1])

                if 'yoff' in ops and ops['yoff'] is not None:
                    ops['yoff'] = ops['yoff'][exp_start_frame:exp_end_frame]
                if 'xoff' in ops and ops['xoff'] is not None:
                    ops['xoff'] = ops['xoff'][exp_start_frame:exp_end_frame]

                if 'yoff1' in ops and ops['yoff1'] is not None:
                    a = np.asarray(ops['yoff1'])
                    if a.ndim == 2:
                        if a.shape[1] == combined_nframes:
                            ops['yoff1'] = a[:, exp_start_frame:exp_end_frame]
                        elif a.shape[0] == combined_nframes:
                            ops['yoff1'] = a[exp_start_frame:exp_end_frame, :]

                if 'xoff1' in ops and ops['xoff1'] is not None:
                    a = np.asarray(ops['xoff1'])
                    if a.ndim == 2:
                        if a.shape[1] == combined_nframes:
                            ops['xoff1'] = a[:, exp_start_frame:exp_end_frame]
                        elif a.shape[0] == combined_nframes:
                            ops['xoff1'] = a[exp_start_frame:exp_end_frame, :]

                np.save(os.path.join(out_plane_dir, 'ops.npy'), ops)
                print('Done updating ops file.')

                print('Cropping and saving binary file (registered frames)...')
                path_to_source_bin = os.path.join(suite2p_combined_path, plane_name, 'data.bin')
                path_to_dest_bin = os.path.join(out_plane_dir, 'data.bin')
                frameSize = combined_ops['meanImg'].shape
                frames_to_copy = range(exp_start_frame, exp_end_frame)

                print('Splitting binary file...')
                split_s2p_vid(path_to_source_bin, path_to_dest_bin, frameSize, frames_to_copy, F.shape[1])
                print('Done splitting binary file.')

                # Per-plane checkpoint right away (fast fail)
                _verify_plane_outputs(exp_dir_processed2, plane_name)

        # permissions
        for iExp in range(len(expIDs)):
            expID2 = expIDs[iExp]
            animalID2, remote_repository_root2, processed_root2, exp_dir_processed2, exp_dir_raw2 = organise_paths.find_paths(userID, expID2)
            if os.path.basename(exp_dir_processed_channel) == 'ch2':
                exp_dir_processed2 = os.path.join(exp_dir_processed2, 'ch2')

            try:
                path = os.path.join(exp_dir_processed2, 'suite2p')
                group_id = grp.getgrnam('users').gr_gid
                mode = 0o770
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        os.chown(dir_path, -1, group_id)
                        os.chmod(dir_path, mode)
                    for f in files:
                        file_path = os.path.join(root, f)
                        os.chown(file_path, -1, group_id)
                        os.chmod(file_path, mode)
            except Exception as e:
                print(f'Problem setting file permissions: {e}')

        # SpinesGUI artifact copy checkpoint
        if copied_map:
            _verify_spinesgui_copies(copied_map)

        # Optional delete, guarded by verification
        if delete_suite2p_combined and os.path.isdir(suite2p_combined_path):
            if verify_before_delete:
                # Re-verify all outputs for all exps/planes before deletion
                for iExp in range(len(expIDs)):
                    expID2 = expIDs[iExp]
                    _, _, _, exp_dir_processed2, _ = organise_paths.find_paths(userID, expID2)
                    if os.path.basename(exp_dir_processed_channel) == 'ch2':
                        exp_dir_processed2 = os.path.join(exp_dir_processed2, 'ch2')
                    for iPlane in range(len(planes_list)):
                        _verify_plane_outputs(exp_dir_processed2, 'plane' + str(iPlane))
                if copied_map:
                    _verify_spinesgui_copies(copied_map)

            print(f"Deleting {suite2p_combined_path!r} ...")
            shutil.rmtree(suite2p_combined_path)


def def_main(
    userID='rubencorreia',
    expID='2025-11-11_01_ESRC022',
    *,
    copy_spinesgui_artifacts=True,
    delete_suite2p_combined=False,
    verify_before_delete=True,
):
    split_combined_suite2p_v3(
        userID,
        expID,
        copy_spinesgui_artifacts=copy_spinesgui_artifacts,
        delete_suite2p_combined=delete_suite2p_combined,
        verify_before_delete=verify_before_delete,
    )


if __name__ == "__main__":
    def_main()
