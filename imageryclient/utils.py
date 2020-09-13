import re
import numpy as np
from functools import partial
from scipy import ndimage
from PIL import Image


def is_precomputed(path):
    if re.search(r'^precomputed://', path) is None:
        return False
    else:
        return True


def _grayscale_to_pil(img, four_channel=False):
    """Helper function to convert one channel uint8 image data to RGB for saving.
    """
    img = img.astype(np.uint8).T
    if four_channel is True:
        sc = 4
    else:
        sc = 3
    pil_img = np.dstack(sc*[img.squeeze()[:, :, np.newaxis]])
    return pil_img


def _binary_mask_to_transparent_pil(img, color=None):
    """Convert a binary array to an MxNx4 RGBa image with fully opaque white (or a specified RGBa color)
    for 1 and fully transparent black for 0.
    """
    if color is None:
        color = [255, 255, 255, 255]
    elif len(color) == 3:
        color = [*color, 255]
    base_img = img.astype(np.uint8).T.squeeze()[:, :, np.newaxis]
    img_r = color[0] * base_img
    img_g = color[1] * base_img
    img_b = color[2] * base_img
    img_a = color[3] * base_img
    pil_img = np.dstack([img_r, img_g, img_b, img_a])
    return pil_img


def _grayscale_to_rgba(img):
    """Convert an rgba 

    Parameters
    ----------
    img : numpy.ndarray
        NxM array of values beteen 0-255.

    Returns
    -------
    numpy.ndarray
        NxMx4 numpy array with the same grayscale colors.
    """
    return _grayscale_to_pil(img)


def convert_to_rgba(img):
    if isinstance(img, Image.Image):
        return img.convert('RGBA')
    else:
        return Image.fromarray(img.T).convert('RGBA')


def convert_to_array(img):
    if isinstance(img, np.ndarray):
        return img
    else:
        return np.array(img).T


def colors_to_uint8(clrs):
    return [[int(255 * ii) for ii in c] for c in clrs]


def binary_seg_outline(seg, width, side='out', color=None, alpha=1):
    """Convert a 2d image segmentation to a binary outline inside or outside the segmentation

    Parameters
    ----------
    seg : PIL image or array
        one-channel PIL Image or 2d array representing image values
    width : int
        Width of outline in pixels
    side : 'out' or 'in', optional
        Whether outline is inside or outside the segmentation mask
    color : list or None, optional
        RGB color for masked values (0-255) or None for white.
    alpha : float, optional
        0-1 value for transparency.

    Returns
    -------
    PIL.Image.Image
        Image with outline
    """
    seg = convert_to_array(seg)
    bseg = seg > 0
    if side == 'out':
        seg_new = ndimage.binary_dilation(
            bseg, iterations=int(width), border_value=0)
        seg_new[bseg] = 0
    else:
        seg_cut = ndimage.binary_erosion(
            bseg, iterations=int(width), border_value=1)
        bseg[seg_cut] = 0
        seg_new = bseg
    if color is None:
        return convert_to_array(seg_new)
    else:
        return colorize_bw(convert_to_array(seg_new), color, new_alpha=alpha)


def mask_image(seg, mask):
    """Apply mask as a a transparency layer to seg

    Parameters
    ----------
    seg : Image.Image
        RGBa image.
    mask : np.ndarray
        Mask with the same number of pixels as the RGBa image. Note that the mask is transposed
        relative to the image pixels.

    Returns
    -------
    Image.Image
        Original segmentation with the mask values set be fully transparent.
    """
    r, g, b, a = np.array(seg).T
    a[mask == 0] = 0
    return Image.fromarray(np.dstack((r.T, g.T, b.T, a.T)))


def colorize_bw(img, new_color, new_alpha=1, background=0, background_alpha=0):
    """Colorize a binary image with a new color, alpha, and background value.

    Parameters
    ----------
    img : PIL.Image.Image
        Image to convert.
    new_color : array-like
        Three element rgb color
    new_alpha : float, optional
        New alpha value, by default 1
    background : float, optional
        background alpha, by default 0

    Returns
    -------
    PIL.Image.Image
        Colorized binary image
    """
    dat = convert_to_rgba(img)
    r, g, b, a = np.array(dat).T
    one_inds = r > 0
    zero_inds = r == 0

    a[zero_inds] = background_alpha
    a[one_inds] = int(new_alpha * 255)

    r[zero_inds] = background
    g[zero_inds] = background
    b[zero_inds] = background

    r[one_inds] = new_color[0]
    g[one_inds] = new_color[1]
    b[one_inds] = new_color[2]
    return Image.fromarray(np.stack([r.T, g.T, b.T, a.T], axis=2).astype('uint8'))


def flatten_data(ii, masks, imagery, dim):
    if imagery is None:
        fimg = None
    else:
        fimg = np.take(imagery, ii, axis=dim)

    if isinstance(masks, dict):
        fmasks = {k: np.take(v, ii, axis=dim) for k, v in masks.items()}
    else:
        fmasks = [np.take(v, ii, axis=dim) for v in masks]
    return fmasks, fimg


def get_first(masks):
    if isinstance(masks, dict):
        return list(masks.values())[0]
    else:
        return masks[0]
