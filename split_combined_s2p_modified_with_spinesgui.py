from xml.etree.ElementPath import ops
import organise_paths
import os
import glob
import numpy as np
import shutil
import grp


# ---------------------------------------------------------------------
# SpinesGUI artifact handling (add-ins)
# ---------------------------------------------------------------------

_SPINESGUI_COPY_GLOBS = [
    "*conversion*.npy",          # conversion libraries
    "extraction_*.txt",          # extraction log(s)
    "*mode*.npy",                # mode file (e.g. ROIs_dendrite_axon_mode.npy)
]

def _list_spinesgui_artifacts(spinesgui_src_dir, extra_globs=None):
    """Return a sorted list of artifact file paths under a SpinesGUI folder."""
    patterns = list(_SPINESGUI_COPY_GLOBS)
    if extra_globs:
        patterns.extend(list(extra_globs))
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(spinesgui_src_dir, pat)))
    # De-dup while keeping deterministic order
    return sorted(set(files))

def copy_spinesgui_artifacts_from_combined_to_split(
    userID,
    expIDs,
    exp_dir_processed_base,
    suite2p_combined_path,
    is_ch2=False,
    verbose=True,
    extra_globs=None,
):
    """
    If suite2p_combined/SpinesGUI exists, copy key SpinesGUI artifacts into each
    split experiment's suite2p/SpinesGUI folder.

    Copies:
      • conversion libraries (*conversion*.npy)
      • extraction log(s) (extraction_*.txt)
      • mode file(s) (*mode*.npy)

    Verification (optional):
      • checks destination files exist and match source file sizes.
    """
    spinesgui_src = os.path.join(suite2p_combined_path, "SpinesGUI")
    if not os.path.isdir(spinesgui_src):
        if verbose:
            print(f"No SpinesGUI folder found in {suite2p_combined_path!r}; skipping artifact copy.")
        return True  # nothing to do, treat as success

    src_files = _list_spinesgui_artifacts(spinesgui_src, extra_globs=extra_globs)
    if len(src_files) == 0:
        if verbose:
            print(f"SpinesGUI folder found at {spinesgui_src!r} but no matching artifacts; skipping.")
        return True

    ok = True
    for expID in expIDs:
        _, _, _, exp_dir_processed2, _ = organise_paths.find_paths(userID, expID)
        if is_ch2:
            exp_dir_processed2 = os.path.join(exp_dir_processed2 + "ch2")

        dest_spinesgui = os.path.join(exp_dir_processed2, "suite2p", "SpinesGUI")
        os.makedirs(dest_spinesgui, exist_ok=True)

        if verbose:
            print(f"Copying SpinesGUI artifacts to {dest_spinesgui!r} ...")

        for src in src_files:
            dst = os.path.join(dest_spinesgui, os.path.basename(src))
            shutil.copy2(src, dst)

            try:
                if not os.path.exists(dst):
                    ok = False
                    if verbose:
                        print(f"  [FAIL] Missing after copy: {dst}")
                else:
                    if os.path.getsize(src) != os.path.getsize(dst):
                        ok = False
                        if verbose:
                            print(f"  [FAIL] Size mismatch: {os.path.basename(src)}")
            except Exception as e:
                ok = False
                if verbose:
                    print(f"  [FAIL] Verification error for {dst}: {e}")

    if verbose:
        print("SpinesGUI artifact copy " + ("OK." if ok else "FAILED."))

    return ok

def delete_suite2p_combined_folder(suite2p_combined_path, verbose=True):
    """
    Delete suite2p_combined folder.
    Caller must ensure it is safe to delete.
    """
    if not os.path.exists(suite2p_combined_path):
        if verbose:
            print(f"{suite2p_combined_path!r} does not exist; nothing to delete.")
        return False

    if verbose:
        print(f"Deleting {suite2p_combined_path!r} ...")

    shutil.rmtree(suite2p_combined_path)

    if verbose:
        print("Deleted suite2p_combined folder.")

    return True

def split_combined_suite2p_v3(userID, expID, *, copy_spinesgui_artifacts=True, delete_combined=False):
    animalID, remote_repository_root, \
        processed_root, exp_dir_processed, \
            exp_dir_raw = organise_paths.find_paths(userID, expID)
    
    # check if two channels have been extracted
    if os.path.exists(os.path.join(exp_dir_processed, 'ch2')):
        # then there are 2 functional channels
        dataPath = [os.path.join(exp_dir_processed), os.path.join(exp_dir_processed, 'ch2')]
    else:
        dataPath = [os.path.join(exp_dir_processed)]

    # interate over channels splitting etc
    for exp_dir_processed in dataPath:


        suite2p_path = os.path.join(exp_dir_processed,'suite2p')
        suite2p_combined_path = os.path.join(exp_dir_processed,'suite2p_combined')
        if not os.path.exists(suite2p_combined_path):
            # Rename the suite2p folder in the first experiment's folder
            os.rename(suite2p_path, suite2p_combined_path)
            #Fixing the ops file on the suite2p_combined folder
            print('Patching ops file from suite2p_combined folder...')
            patch_all_ops_paths(userID, expID)
            print('Done patching ops file.')

        planes_list = glob.glob(os.path.join(suite2p_combined_path, '*plane*'))
        # determine all experiment IDs that have been combined
        combined_ops = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane0','ops.npy'),allow_pickle = True).item()
        iscell = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane0','iscell.npy'))

        expIDs = {}
        for iExp in range(len(combined_ops['data_path'])):
            expIDs[iExp] = os.path.basename(combined_ops['data_path'][iExp])

        all_animal_ids = []
        # check all experiments from the same animal
        for iExp in range(len(combined_ops['data_path'])):
            all_animal_ids.append(expIDs[iExp][14:])

        if len(set(all_animal_ids)) > 1:
            raise Exception('Combined multiple animals not permitted')

        for iPlane in range(len(planes_list)):
            print('Plane ' + str(iPlane))
            # load the combined data
            F = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'F.npy'))
            Fneu = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'Fneu.npy'))
            spks = np.load(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'spks.npy'))
            # iterate through experiments grabbing each's frames
            for iExp in range(len(expIDs)):
                expID = expIDs[iExp]
                frames_in_exp = combined_ops['frames_per_folder'][iExp]
                # calculate which frame in the combined data is the first from this experiment
                exp_start_frame = np.sum(combined_ops['frames_per_folder'][0:iExp]).astype(int)
                # define which frame is the last frame from this experiment
                exp_end_frame = exp_start_frame + frames_in_exp - 1
                # select frames that come from this experiment
                F_exp = F[:,exp_start_frame:exp_end_frame]
                Fneu_exp = Fneu[:,exp_start_frame:exp_end_frame]
                spks_exp = spks[:,exp_start_frame:exp_end_frame]
                # save to experiment directory
                animalID2, remote_repository_root2, \
                    processed_root2, exp_dir_processed2, \
                        exp_dir_raw2 = organise_paths.find_paths(userID, expID)
                if exp_dir_processed[-3:] == 'ch2':
                    # then we are splitting ch2
                    exp_dir_processed2 = os.path.join(exp_dir_processed2 + 'ch2')
                # make output directory if it doesn't already exist
                os.makedirs(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane)), exist_ok = True)
                # save appropriate part of F etc
                print('Cropping and saving cell traces...')
                np.save(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'F.npy'),F_exp)
                np.save(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'Fneu.npy'),Fneu_exp)
                np.save(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'spks.npy'),spks_exp)
                # copy across iscell and ops files
                shutil.copy(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'iscell.npy'), \
                            os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'iscell.npy'))
                shutil.copy(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'stat.npy'), \
                            os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'stat.npy'))
                shutil.copy(os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'ops.npy'), \
                            os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'ops.npy'))
                # copy frames from registered video bin file to split folder
                print('Cropping and saving binary file (registered frames)...')
                path_to_source_bin = os.path.join(exp_dir_processed,'suite2p_combined','plane'+str(iPlane),'data.bin')
                path_to_dest_bin = os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'data.bin')
                frameSize = combined_ops['meanImg'].shape
                frames_to_copy = range(exp_start_frame,exp_end_frame)

                #Changing the paths and frame details in the splited ops.file
                print('Updating ops file...')
                ops = np.load(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'ops.npy'),allow_pickle = True).item()
                ops['nframes'] = frames_in_exp
                ops['ops_path'] = os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'ops.npy')
                ops['reg_file'] = os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'data.bin')
                ops['frames_per_folder'] = [frames_in_exp]
                ops.pop('reg_file_chan2', None) #Needed to avoid problems with the GUI (It will try to load a file that was not moved into the folder)

                #Filtering out frames_per_file that are not needed
                files = ops['filelist']
                frames_per_file = ops['frames_per_file']
                keep_mask = [expID in f for f in files]
                filtered_files = [f for f, keep in zip(files, keep_mask) if keep]
                filtered_frames = [n for n, keep in zip(frames_per_file, keep_mask) if keep]
                ops['filelist'] = filtered_files
                ops['frames_per_file'] = filtered_frames
                ops['nframes'] = sum(filtered_frames)

                #Slice motion offsets to the experiment frames
                combined_nframes = F.shape[1]  # frames in the combined run

                # rigid (1D)
                if 'yoff' in ops and ops['yoff'] is not None:
                    ops['yoff'] = ops['yoff'][exp_start_frame:exp_end_frame]

                if 'xoff' in ops and ops['xoff'] is not None:
                    ops['xoff'] = ops['xoff'][exp_start_frame:exp_end_frame]

                # non-rigid (2D) — detect which axis is frames
                if 'yoff1' in ops and ops['yoff1'] is not None:
                    a = np.asarray(ops['yoff1'])
                    if a.ndim == 2:
                        if a.shape[1] == combined_nframes:      # (blocks, frames)
                            ops['yoff1'] = a[:, exp_start_frame:exp_end_frame]
                        elif a.shape[0] == combined_nframes:    # (frames, blocks)
                            ops['yoff1'] = a[exp_start_frame:exp_end_frame, :]

                if 'xoff1' in ops and ops['xoff1'] is not None:
                    a = np.asarray(ops['xoff1'])
                    if a.ndim == 2:
                        if a.shape[1] == combined_nframes:
                            ops['xoff1'] = a[:, exp_start_frame:exp_end_frame]
                        elif a.shape[0] == combined_nframes:
                            ops['xoff1'] = a[exp_start_frame:exp_end_frame, :]

                #Saving the updated ops file
                np.save(os.path.join(exp_dir_processed2,'suite2p','plane'+str(iPlane),'ops.npy'),ops)
                print('Done updating ops file.')

                # call the function to split
                print('Splitting binary file...')
                split_s2p_vid(path_to_source_bin,path_to_dest_bin,frameSize,frames_to_copy,F.shape[1]);
                print('Done splitting binary file.')

                # sort out permissions
            for iExp in range(len(expIDs)):
                expID = expIDs[iExp]
                animalID2, remote_repository_root2, \
                    processed_root2, exp_dir_processed2, \
                        exp_dir_raw2 = organise_paths.find_paths(userID, expID)
                try:
                    # animalID, remote_repository_root, processed_root, exp_dir_processed, exp_dir_raw = organise_paths.find_paths(userID, expID)
                    path = os.path.join(exp_dir_processed2,'suite2p')
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
                except:
                    print('Problem setting file permissions to user in step 1 batch')

        # -------------------------------------------------------------
        # Optional: propagate SpinesGUI artifacts from suite2p_combined
        # -------------------------------------------------------------
        # After splitting is done for this channel, copy SpinesGUI artifacts (if any)
        verified_ok = True
        if copy_spinesgui_artifacts:
            verified_ok = copy_spinesgui_artifacts_from_combined_to_split(
                userID=userID,
                expIDs=list(expIDs.values()),
                exp_dir_processed_base=exp_dir_processed,
                suite2p_combined_path=suite2p_combined_path,
                is_ch2=(exp_dir_processed[-3:] == 'ch2'),
                verbose=True,
            )

        # ---------------------------------------------------------------------
        # Optional: delete suite2p_combined after successful split
        # (verification is mandatory, so deletion is only allowed if verified_ok)
        # ---------------------------------------------------------------------
        if delete_combined:
            if not verified_ok:
                raise RuntimeError(
                    "SpinesGUI verification failed (verified_ok=False) — refusing to delete suite2p_combined."
                )

            delete_suite2p_combined_folder(
                suite2p_combined_path,
                verbose=True,
            )




def split_s2p_vid(path_to_source_bin, path_to_dest_bin, frameSize, frames_to_copy,total_frames):
    frames_to_copy = np.array(frames_to_copy, dtype=float)
    blockSize = 1000

    finfo = os.stat(path_to_source_bin)
    fsize = finfo.st_size
    frameCountCalculation = fsize / (frameSize[0] * frameSize[1] * 2)

    total_frames_to_write = len(frames_to_copy)

    frame_mean = []
    framesInSet = []

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
            print('Reading...')
            read_data = np.fromfile(fid, dtype=np.int16, count=frameSize[0]*frameSize[1]*framesToRead)

            # write to other file
            print('Writing...')
            read_data.tofile(fid2)

            # debug
            if iStart == 1:
                combined_data = read_data.reshape((frameSize[0], frameSize[1], framesToRead))

    combined_data = np.squeeze(np.mean(combined_data, axis=2))
    return combined_data

def patch_all_ops_paths(userID, expID):
    """
    Recursively find every ops.npy under suite2p/ and:
      - reset ops['ops_path'] to its own file path
      - rebase reg_file and reg_file_chan2 if they exist
      - save back out
    """
    _, _, _, exp_dir_processed, _ = organise_paths.find_paths(userID, expID)
    suite2p_folder = os.path.join(exp_dir_processed, 'suite2p_combined')

    # find every ops.npy
    pattern = os.path.join(suite2p_folder, '**', 'ops.npy')
    ops_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(ops_files)} ops.npy files under {suite2p_folder!r}")

    for ops_path in ops_files:
        folder = os.path.dirname(ops_path)
        #print(f"Patching {ops_path!r}…")
        ops = np.load(ops_path, allow_pickle=True).item()

        # reset the ops_path
        ops['ops_path'] = ops_path

        # rebase any registration file paths
        for key in ('reg_file', 'reg_file_chan2'):
            if key in ops:
                old = ops[key]
                new = os.path.join(folder, os.path.basename(old))
                ops[key] = new
                print(f"  • {key}: {old!r} → {new!r}")

        np.save(ops_path, ops)
        #print(f"  ✔ saved patched ops.npy")

    print("All ops.npy files have been updated.")

if __name__ == "__main__":
    userID = 'rubencorreia'
    expID  = '2025-11-11_01_ESRC022'    # <--- put the first experiment of the sequence here
    split_combined_suite2p_v3(userID, expID, copy_spinesgui_artifacts=True, verify_spinesgui_copy=True, delete_combined=False)