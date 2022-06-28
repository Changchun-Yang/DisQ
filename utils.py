# internal python imports
import os
import csv
import functools

# third party imports
import numpy as np
import scipy
from skimage import measure
import skimage
import scipy.io as scio
import torch
import scipy.stats as stats
import pdb
import shutil
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import nn
import math


# local/our imports
import pystrum.pynd.ndutils as nd


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'tensorflow' if os.environ.get('VXM_BACKEND') == 'tensorflow' else 'pytorch'


def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def read_mat_list(filename, train=True):
    '''
    Reads a list of files from a mat folder.

    Parameters:
        filename: the mat data path.
    '''
    if train:
        data_path = os.path.join(filename, "Train/")
    else:
        data_path = os.path.join(filename, "Val/")
    filelist = []
    for f in sorted(os.listdir(data_path)):
        filelist.append(os.path.join(data_path, f))
    return filelist


def read_cmat_list(filename, data_name, train=True):
    '''
    Reads a list of files from a contrast mat folder.

    Parameters:
        filename: the mat data path.
    '''
    if train:
        data_path = os.path.join(filename, "train_%s/" % data_name)
    else:
        data_path = os.path.join(filename, "val_%s/" % data_name)
    filelist = []
    for f in sorted(os.listdir(data_path)):
        filelist.append(os.path.join(data_path, f))
    return filelist


def read_pair_list(filename, delim=None, prefix=None, suffix=None):
    '''
    Reads a list of registration file pairs from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        delim: File pair delimiter. Default is a whitespace seperator (None).
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    pairlist = [f.split(delim) for f in read_file_list(filename)]
    if prefix is not None:
        pairlist = [[prefix + f for f in pair] for pair in pairlist]
    if suffix is not None:
        pairlist = [[f + suffix for f in pair] for pair in pairlist]
    return pairlist


def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data().squeeze()
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol


def crop_and_fill(img, size):
    img_new = np.zeros((img.shape[0], img.shape[1], size, size))
    h = np.amin([size, img.shape[2]])
    w = np.amin([size, img.shape[3]])
    img_new[:, :, size // 2 - h // 2:size // 2 + h // 2, size // 2 - w // 2:size // 2 + w // 2] = \
        img[:, :, img.shape[2] // 2 - h // 2:img.shape[2] // 2 + h // 2, img.shape[3] // 2 - w // 2:img.shape[3] // 2 + w // 2]
    return img_new

def crop_and_fill2(img, size):
    img_new = np.zeros((size, size))
    h = np.amin([size, img.shape[0]])
    w = np.amin([size, img.shape[1]])
    img_new[size // 2 - h // 2:size // 2 + h // 2, size // 2 - w // 2:size // 2 + w // 2] = \
        img[img.shape[0] // 2 - h // 2:img.shape[0] // 2 + h // 2, img.shape[1] // 2 - w // 2:img.shape[1] // 2 + w // 2]
    return img_new


def get_max_min(data_path, np_var='volume'):

    data_path_train = os.path.join(data_path, "Train/")

    data_path_Val = os.path.join(data_path, "Val/")

    max_list = []
    min_list = []
    for f in sorted(os.listdir(data_path_train)):
        filename = os.path.join(data_path_train, f)
        mat = scio.loadmat(filename)
        vol = mat[np_var]
        max_list.append(np.max(vol))
        min_list.append(np.min(vol))

    ## val, we can not obtain the max and min of test data
    for f in sorted(os.listdir(data_path_Val)):
        filename = os.path.join(data_path_Val, f)
        mat = scio.loadmat(filename)
        vol = mat[np_var]
        max_list.append(np.max(vol))
        min_list.append(np.min(vol))

    return np.max(max_list), np.min(min_list)


def load_matfile(
        filename,
        np_var='vol',
        add_batch_axis=False,
        add_feat_axis=False,
        pad_shape=None,
        resize_factor=1,
        ret_affine=False
):

    mat = scio.loadmat(filename)
    vol = mat[np_var]

    vol = np.transpose(vol, (2, 3, 0, 1))

    if pad_shape:
        vol = crop_and_fill(vol, pad_shape)

    # vol_max = np.max(np.abs(vol))
    # vol /= vol_max

    scan_num = vol.shape[0]
    slice_num = vol.shape[1]
    pair = 2

    ## Scan to scan
    # ind_scan = np.random.randint(scan_num, size=pair)
    # ind_slice = np.random.randint(slice_num, size=pair)
    # slices = [vol[ind_scan[i]][ind_slice[i]] for i in range(pair)]

    # Slice to slice
    ind_scan = np.random.randint(scan_num, size=1)
    # ind_slice = np.random.randint(slice_num, size=pair)
    ind_slice = np.random.choice(range(slice_num), pair, replace=False)  # no repetitive

    sel_vol = vol[ind_scan[0]]

    # # Get the threshold of 98% pixels of the histogram
    # arr = sel_vol.flatten()
    # n, bins, patches = plt.hist(arr, bins=int(np.max(arr)), cumulative=True)
    # l_d = np.argwhere(n >= arr.shape[0] * 0.98)
    percent_index = np.percentile(sel_vol, 98)

    # Normalize to 0~1
    slices = [np.clip(sel_vol[ind_slice[i]], 0, percent_index - 1) / (percent_index - 1) for i in range(pair)]

    if add_feat_axis:
        for i in range(pair):
            slices[i] = slices[i][np.newaxis, ...]

    slices = np.concatenate(slices, axis=0)

    if add_batch_axis:
        slices = slices[np.newaxis, ...]
    slices = np.array(slices, dtype='float32')
    return slices


def load_mat_gen_file(
        filename,
        np_var='vol',
        add_batch_axis=False,
        add_feat_axis=False,
        pad_shape=None,
        resize_factor=1,
        ret_affine=False
):

    mat = scio.loadmat(filename)
    vol = mat[np_var]

    vol = np.transpose(vol, (2, 0, 1))

    # vol_max = np.max(np.abs(vol))
    # vol /= vol_max

    slice_num = vol.shape[0]
    pair = 2

    ## Scan to scan
    # ind_scan = np.random.randint(scan_num, size=pair)
    # ind_slice = np.random.randint(slice_num, size=pair)
    # slices = [vol[ind_scan[i]][ind_slice[i]] for i in range(pair)]

    # Slice to slice
    # ind_slice = np.random.randint(slice_num, size=pair)
    ind_slice = np.random.choice(range(slice_num), pair, replace=False)  # no repetitive

    sel_vol = vol[ind_slice]

    # # Get the threshold of 98% pixels of the histogram
    # arr = sel_vol.flatten()
    # n, bins, patches = plt.hist(arr, bins=int(np.max(arr)), cumulative=True)
    # l_d = np.argwhere(n >= arr.shape[0] * 0.98)
    percent_index = np.percentile(sel_vol, 100)

    # Normalize to 0~1
    slices = [np.clip(sel_vol[i], 0, percent_index - 1) / (percent_index - 1) for i in range(pair)]

    if add_feat_axis:
        for i in range(pair):

            slices_i = slices[i]
            slices_i = slices_i[~(slices_i==0).all(1), :]
            slices_i = slices_i[:, ~(slices_i==0).all(0)]
            slices_i = crop_and_fill2(slices_i, pad_shape)
            slices[i] = slices_i[np.newaxis, ...]

    slices = np.concatenate(slices, axis=0)

    if add_batch_axis:
        slices = slices[np.newaxis, ...]
    slices = np.array(slices, dtype='float32')
    return slices


def contrast_load_matfile(
        filename,
        np_var='vol',
        add_batch_axis=False,
        add_feat_axis=False,
        pad_shape=None,
        resize_factor=1,
        ret_affine=False
):

    mat = scio.loadmat(filename)
    vol = mat[np_var]
    vol_sc = mat[np_var + '_sc']

    vol = np.transpose(vol, (2, 3, 0, 1))
    vol_sc = np.transpose(vol_sc, (2, 3, 0, 1))

    # already pad
    # if pad_shape:
    #     vol = crop_and_fill(vol, pad_shape)

    # vol_max = np.max(np.abs(vol))
    # vol /= vol_max

    scan_num = vol.shape[0]
    slice_num = vol.shape[1]
    pair = 2

    ## Scan to scan
    # ind_scan = np.random.randint(scan_num, size=pair)
    # ind_slice = np.random.randint(slice_num, size=pair)
    # slices = [vol[ind_scan[i]][ind_slice[i]] for i in range(pair)]

    # Slice to slice
    ind_scan = np.random.randint(scan_num, size=1)
    # ind_slice = np.random.randint(slice_num, size=pair)
    ind_slice = np.random.choice(range(slice_num), pair, replace=False)  # no repetitive

    sel_vol = vol[ind_scan[0]]
    sel_vol_sc = vol_sc[ind_scan[0]]

    # already processed
    # # # Get the threshold of 98% pixels of the histogram
    # # arr = sel_vol.flatten()
    # # n, bins, patches = plt.hist(arr, bins=int(np.max(arr)), cumulative=True)
    # # l_d = np.argwhere(n >= arr.shape[0] * 0.98)
    # percent_index = np.percentile(sel_vol, 98)

    # Normalize to 0~1
    #slices = [np.clip(sel_vol[ind_slice[i]], 0, percent_index - 1) / (percent_index - 1) for i in range(pair)]
    max_num = np.max(sel_vol)
    slices = [sel_vol[ind_slice[i]] / max_num for i in range(pair)]
    slices_sc = [sel_vol_sc[ind_slice[i]] / max_num for i in range(pair)]

    if add_feat_axis:
        for i in range(pair):
            slices[i] = slices[i][np.newaxis, ...]
            slices_sc[i] = slices_sc[i][np.newaxis, ...]

    slices = np.concatenate(slices, axis=0)
    slices_sc = np.concatenate(slices_sc, axis=0)

    if add_batch_axis:
        slices = slices[np.newaxis, ...]
        slices_sc = slices_sc[np.newaxis, ...]
    slices = np.array(slices, dtype='float32')
    slices_sc = np.array(slices_sc, dtype='float32')
    return slices, slices_sc


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def load_labels(arg):
    """
    Load label maps and return a list of unique labels as well as all maps.

    Parameters:
        arg: Path to folder containing label maps, string for globbing, or a list of these.

    Returns:
        np.array: List of unique labels.
        list: List of label maps, each as a np.array.
    """
    if not isinstance(arg, (tuple, list)):
        arg = [arg]

    # List files.
    import glob
    ext = ('.nii.gz', '.nii', '.mgz', '.npy', '.npz')
    files = [os.path.join(f, '*') if os.path.isdir(f) else f for f in arg]
    files = sum((glob.glob(f) for f in files), [])
    files = [f for f in files if f.endswith(ext)]

    # Load labels.
    if len(files) == 0:
        raise ValueError(f'no labels found for argument "{files}"')
    label_maps = []
    shape = None
    for f in files:
        x = np.squeeze(load_volfile(f))
        if shape is None:
            shape = np.shape(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f'file "{f}" has non-integral data type')
        if not np.all(x.shape == shape):
            raise ValueError(f'shape {x.shape} of file "{f}" is not {shape}')
        label_maps.append(x)

    return np.unique(label_maps), label_maps


def load_pheno_csv(filename, training_files=None):
    """
    Loads an attribute csv file into a dictionary. Each line in the csv should represent
    attributes for a single training file and should be formatted as:

    filename,attr1,attr2,attr2...

    Where filename is the file basename and each attr is a floating point number. If
    a list of training_files is specified, the dictionary file keys will be updated
    to match the paths specified in the list. Any training files not found in the
    loaded dictionary are pruned.
    """

    # load csv into dictionary
    pheno = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            pheno[row[0]] = np.array([float(f) for f in row[1:]])

    # make list of valid training files
    if training_files is None:
        training_files = list(training_files.keys())
    else:
        training_files = [f for f in training_files if os.path.basename(f) in pheno.keys()]
        # make sure pheno dictionary includes the correct path to training data
        for f in training_files:
            pheno[f] = pheno[os.path.basename(f)]

    return pheno, training_files


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if not batch_axis:
            dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def affine_shift_to_matrix(trf, resize=None, unshift_shape=None):
    """
    Converts an affine shift to a matrix (over the identity).
    To convert back from center-shifted transform, provide image shape
    to unshift_shape.

    TODO: make ND compatible - currently just 3D
    """
    matrix = np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)
    if resize is not None:
        matrix[:3, -1] *= resize
    if unshift_shape is not None:
        T = np.zeros((4, 4))
        T[:3, 3] = (np.array(unshift_shape) - 1) / 2
        matrix = (np.eye(4) + T) @ matrix @ (np.eye(4) - T)
    return matrix


def extract_largest_vol(bw, connectivity=1):
    """
    Extracts the binary (boolean) image with just the largest component.
    TODO: This might be less than efficiently implemented.
    """
    lab = measure.label(bw.astype('int'), connectivity=connectivity)
    regions = measure.regionprops(lab, cache=False)
    areas = [f.area for f in regions]
    ai = np.argsort(areas)[::-1]
    bw = lab == ai[0] + 1
    return bw


def clean_seg(x, std=1):
    """
    Cleans a segmentation image.
    """

    # take out islands, fill in holes, and gaussian blur
    bw = extract_largest_vol(x)
    bw = 1 - extract_largest_vol(1 - bw)
    gadt = scipy.ndimage.gaussian_filter(bw.astype('float'), std)

    # figure out the proper threshold to maintain the total volume
    sgadt = np.sort(gadt.flatten())[::-1]
    thr = sgadt[np.ceil(bw.sum()).astype(int)]
    clean_bw = gadt > thr

    assert np.isclose(bw.sum(), clean_bw.sum(), atol=5), 'cleaning segmentation failed'
    return clean_bw.astype(float)


def clean_seg_batch(X_label, std=1):
    """
    Cleans batches of segmentation images.
    """
    if not X_label.dtype == 'float':
        X_label = X_label.astype('float')

    data = np.zeros(X_label.shape)
    for xi, x in enumerate(X_label):
        data[xi, ..., 0] = clean_seg(x[..., 0], std)

    return data


def filter_labels(atlas_vol, labels):
    """
    Filters given volumes to only include given labels, all other voxels are set to 0.
    """
    mask = np.zeros(atlas_vol.shape, 'bool')
    for label in labels:
        mask = np.logical_or(mask, atlas_vol == label)
    return atlas_vol * mask


def dist_trf(bwvol):
    """
    Computes positive distance transform from positive entries in a logical image.
    """
    revbwvol = np.logical_not(bwvol)
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)


def signed_dist_trf(bwvol):
    """
    Computes the signed distance transform from the surface between the binary
    elements of an image
    NOTE: The distance transform on either side of the surface will be +/- 1,
    so there are no voxels for which the distance should be 0.
    NOTE: Currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    """

    # get the positive transform (outside the positive island)
    posdst = dist_trf(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = dist_trf(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def vol_to_sdt(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transform from a volume.
    """

    X_dt = signed_dist_trf(X_label)

    if not (sdt_vol_resize == 1):
        if not isinstance(sdt_vol_resize, (list, tuple)):
            sdt_vol_resize = [sdt_vol_resize] * X_dt.ndim
        if any([f != 1 for f in sdt_vol_resize]):
            X_dt = scipy.ndimage.interpolation.zoom(X_dt, sdt_vol_resize, order=1, mode='reflect')

    if not sdt:
        X_dt = np.abs(X_dt)

    return X_dt


def vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transforms from volume batches.
    """

    # assume X_label is [batch_size, *vol_shape, 1]
    assert X_label.shape[-1] == 1, 'implemented assuming size is [batch_size, *vol_shape, 1]'
    X_lst = [f[..., 0] for f in X_label]  # get rows
    X_dt_lst = [vol_to_sdt(f, sdt=sdt, sdt_vol_resize=sdt_vol_resize)
                for f in X_lst]  # distance transform
    X_dt = np.stack(X_dt_lst, 0)[..., np.newaxis]
    return X_dt


def get_surface_pts_per_label(total_nb_surface_pts, layer_edge_ratios):
    """
    Gets the number of surface points per label, given the total number of surface points.
    """
    nb_surface_pts_sel = np.round(np.array(layer_edge_ratios) * total_nb_surface_pts).astype('int')
    nb_surface_pts_sel[-1] = total_nb_surface_pts - int(np.sum(nb_surface_pts_sel[:-1]))
    return nb_surface_pts_sel


def edge_to_surface_pts(X_edges, nb_surface_pts=None):
    """
    Converts edges to surface points.
    """

    # assumes X_edges is NOT in keras form
    surface_pts = np.stack(np.where(X_edges), 0).transpose()

    # random with replacements
    if nb_surface_pts is not None:
        chi = np.random.choice(range(surface_pts.shape[0]), size=nb_surface_pts)
        surface_pts = surface_pts[chi, :]

    return surface_pts


def sdt_to_surface_pts(X_sdt, nb_surface_pts,
                       surface_pts_upsample_factor=2, thr=0.50001, resize_fn=None):
    """
    Converts a signed distance transform to surface points.
    """
    us = [surface_pts_upsample_factor] * X_sdt.ndim

    if resize_fn is None:
        resized_vol = scipy.ndimage.interpolation.zoom(X_sdt, us, order=1, mode='reflect')
    else:
        resized_vol = resize_fn(X_sdt)
        pred_shape = np.array(X_sdt.shape) * surface_pts_upsample_factor
        assert np.array_equal(pred_shape, resized_vol.shape), 'resizing failed'

    X_edges = np.abs(resized_vol) < thr
    sf_pts = edge_to_surface_pts(X_edges, nb_surface_pts=nb_surface_pts)

    # can't just correct by surface_pts_upsample_factor because of how interpolation works...
    pt = [sf_pts[..., f] * (X_sdt.shape[f] - 1) / (X_edges.shape[f] - 1) for f in range(X_sdt.ndim)]
    return np.stack(pt, -1)


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


# load model
def load_checkpoint_by_key(values, checkpoint_dir, keys, device, ckpt_name='model_best.pth.tar'):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = os.path.join(checkpoint_dir, ckpt_name)
    print(filename)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            try:
                if key == 'model':
                    values[i] = load_checkpoint_model(values[i], checkpoint[key])
                else:
                    values[i].load_state_dict(checkpoint[key])
                print('loading ' + key + ' success!')
            except:
                print('loading ' + key + ' failed!')
        # print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(filename, epoch, checkpoint['monitor_metric']))
        print("loaded checkpoint from '{}' (epoch: {})".format(filename, epoch))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch


def load_regcheckpoint_by_key(values, checkpoint_dir, keys, device, ckpt_name='model_best.pth.tar'):
    filename = os.path.join(checkpoint_dir, ckpt_name)
    print(filename)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            try:
                if 'model' in key:
                    values[i] = load_checkpoint_model(values[i], checkpoint[key])
                else:
                    values[i].load_state_dict(checkpoint[key])
                print('loading ' + key + ' success!')
            except:
                print('loading ' + key + ' failed!')
        # print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(filename, epoch, checkpoint['monitor_metric']))
        print("loaded checkpoint from '{}' (epoch: {})".format(filename, epoch))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch


def load_checkpoint_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def save_config_file(args):
    file_path = os.path.join(args.ckpt_path, 'args.txt')

    f = open(file_path, 'w')
    # for key, value in args.items():
    #     f.write(key + ': ' + str(value) + '\n')
    f.close()

def compute_reconstruction_metrics_single(target, pred):
    # target = target / target.max() + 1e-8
    # pred = pred / pred.max() + 1e-8
    # range = np.max(target) - np.min(target)
    target = target - target.min()
    pred = pred - pred.min()
    range = target.max()
    try:
        rmse_pred = skimage.metrics.mean_squared_error(target, pred)
        # rmse_pred = skimage.metrics.normalized_root_mse(target, pred)
    except:
        rmse_pred = float('nan')
    try:
        # psnr_pred = skimage.metrics.peak_signal_noise_ratio(target, pred)
        psnr_pred = skimage.metrics.peak_signal_noise_ratio(target, pred, data_range=range)
    except:
        pdb.set_trace()
        psnr_pred = float('nan')
    try:
        # ssim_pred = skimage.metrics.structural_similarity(target, pred)
        ssim_pred = skimage.metrics.structural_similarity(target, pred, data_range=range)
    except:
        ssim_pred = float('nan')
    return {'ssim': ssim_pred, 'rmse': rmse_pred, 'psnr': psnr_pred}


def compute_reconstruction_metrics(target, pred):
    ssim_list = []
    rmse_list = []
    psnr_list = []
    for i in range(target.shape[0]):
        metrics_dict = compute_reconstruction_metrics_single(target[i,0], pred[i,0])
        ssim_list.append(metrics_dict['ssim'])
        psnr_list.append(metrics_dict['psnr'])
        rmse_list.append(metrics_dict['rmse'])
    return {'ssim': ssim_list, 'psnr': psnr_list, 'rmse': rmse_list}


def save_checkpoint(state, is_best, checkpoint_dir):
    # print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')

# http://stackoverflow.com/a/22718321
def mkdir_p(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def softmax(logits, temperature=1.0, dim=1):
    exps = torch.exp(logits/temperature)
    return exps/torch.sum(exps, dim=dim)

def create_one_hot(soft_prob, dim):
    indices = torch.argmax(soft_prob, dim=dim)
    hard = F.one_hot(indices, soft_prob.size()[dim])
    new_axes = tuple(range(dim)) + (len(hard.shape)-1,) + tuple(range(dim, len(hard.shape)-1))
    return hard.permute(new_axes).float()

class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
    def forward(self, mu, logvar):
        kld = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)) - 0.5
        if self.reduction == 'mean':
            kld = kld.mean()
        return kld

class ShapeLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(ShapeLoss, self).__init__()
        self.reduction = reduction
    def forward(self, x, y):
        # x = F.normalize(x, dim=1, p=2)
        # y = F.normalize(y, dim=1, p=2)
        #return 2 - 2 * (x * y).sum(dim=-1)
        return 2 - 2 * (x * y)


class SmoothLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(SmoothLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        dx = dx*dx
        dy = dy*dy
        d = torch.mean(dx) + torch.mean(dy)
        grad = d
        return d


class NCCLoss(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCCLoss, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class TemperatureAnneal:
    def __init__(self, initial_temp=1.0, anneal_rate=0.0, min_temp=0.5, device=torch.device('cuda')):
        self.initial_temp = initial_temp
        self.anneal_rate = anneal_rate
        self.min_temp = min_temp
        self.device = device

        self._temperature = self.initial_temp
        self.last_epoch = 0

    def get_temp(self):
        return self._temperature

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        current_temp = self.initial_temp * np.exp(-self.anneal_rate * self.last_epoch)
        # noinspection PyArgumentList
        self._temperature = torch.max(torch.FloatTensor([current_temp, self.min_temp]).to(self.device))

    def reset(self):
        self._temperature = self.initial_temp
        self.last_epoch = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
