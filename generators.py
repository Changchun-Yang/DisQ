import os
import sys
import glob
import numpy as np
import utils

def volgen(
    vol_names,
    batch_size=1,
    segs=None,
    np_var='vol',
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern, a list of file paths, or a list of
    preloaded volumes. Corresponding segmentations are additionally loaded if
    `segs` is provided as a list (of file paths or preloaded segmentations) or set
    to True. If `segs` is True, npz files with variable names 'vol' and 'seg' are
    expected. Passing in preloaded volumes (with optional preloaded segmentations)
    allows volumes preloaded in memory to be passed to a generator.

    Parameters:
        vol_names: Path, glob pattern, list of volume files to load, or list of
            preloaded volumes.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        imgs = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if segs is True:
            # assume inputs are npz files with 'seg' key
            load_params['np_var'] = 'seg'  # be sure to load seg
            s = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(s, axis=0))
        elif isinstance(segs, list):
            # assume segs is a corresponding list of files or preloaded volumes
            s = [utils.load_volfile(segs[i], **load_params) for i in indices]
            vols.append(np.concatenate(s, axis=0))

        yield tuple(vols)


def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). 
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)


def matgen(
        vol_names,
        batch_size=1,
        segs=None,
        np_var='volume',
        pad_shape=192,
        resize_factor=1,
        add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern, a list of file paths, or a list of
    preloaded volumes. Corresponding segmentations are additionally loaded if
    `segs` is provided as a list (of file paths or preloaded segmentations) or set
    to True. If `segs` is True, npz files with variable names 'vol' and 'seg' are
    expected. Passing in preloaded volumes (with optional preloaded segmentations)
    allows volumes preloaded in memory to be passed to a generator.

    Parameters:
        vol_names: Path, glob pattern, list of volume files to load, or list of
            preloaded volumes.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        imgs = [utils.load_matfile(vol_names[i], **load_params) for i in indices]  # t1mapping data  "volume" pad_shape=256
        # imgs = [utils.load_mat_gen_file(vol_names[i], **load_params) for i in indices]  # gen data   "volume_gen" pad_shape=192
        vols = [np.concatenate(imgs, axis=0)]

        yield tuple(vols)


def slice_to_slice(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for slice-to-slice registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training).
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = matgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan = next(gen)[0]
        scan1 = scan[:, 0:1, :, :]
        scan2 = scan[:, 1:2, :, :]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[2:]
            zeros = np.zeros((batch_size,  len(shape), *shape))

        invols = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)


def contrast_matgen(
        vol_names,
        batch_size=1,
        segs=None,
        np_var='volume',
        pad_shape=256,
        resize_factor=1,
        add_feat_axis=True
):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        imgs = [utils.contrast_load_matfile(vol_names[i], **load_params) for i in indices]

        vols = []
        vols_sc = []
        for i in range(batch_size):
            vols.append(imgs[i][0])
            vols_sc.append(imgs[i][1])
        vols = np.concatenate(vols, axis=0)
        vols_sc = np.concatenate(vols_sc, axis=0)
        yield tuple([vols, vols_sc])


def contrast_slice_to_slice(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    zeros = None
    gen = contrast_matgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan, scan_sc = next(gen)
        scan1 = scan[:, 0:1, :, :]
        scan2 = scan[:, 1:2, :, :]

        scan_sc1 = scan_sc[:, 0:1, :, :]
        scan_sc2 = scan_sc[:, 1:2, :, :]

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[2:]
            zeros = np.zeros((batch_size,  len(shape), *shape))
            zeros_sc = zeros

        invols = [scan1, scan2]
        invols_sc = [scan_sc1, scan_sc2]

        outvols = [scan2, scan1] if bidir else [scan2]
        outvols_sc = [scan_sc2, scan_sc1] if bidir else [scan_sc2]
        if not no_warp:
            outvols.append(zeros)
            outvols_sc.append(zeros_sc)

        yield (invols, outvols, invols_sc, outvols_sc)
