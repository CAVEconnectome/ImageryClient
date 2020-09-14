import numpy as np
from PIL import Image
from . import utils
from seaborn import husl_palette, hls_palette, color_palette

DEFAULT_PALETTE = 'husl'
DEFAULT_H = 0.01
DEFAULT_L = 0.65
DEFAULT_S = 1.0


def discrete_colors(segs,
                    palette=DEFAULT_PALETTE,
                    h=DEFAULT_H,
                    l=DEFAULT_L,
                    s=DEFAULT_S,
                    ):
    """Generate discrete colors for segmentations from a palette
    generator. Defaults to perceptually uniform differences with
    high saturation.

    Parameters
    ----------
    segs : list or dict
        Dict or list of segmentations to provide colors for.
    palette : 'husl', 'hls', or str, optional
        Which palette system to use, by default 'husl'. Will
        accept anything allowed by seaborn color_palette function.
    h : float, optional
        Hue value if husl or hls palettes are used, by default 0.01
    l : float, optional
        Lightness if husl or hls palettes are used, by default 0.6
    s : int, optional
        Saturation if husl or hls palettes are used, by default 1

    Returns
    -------
    List or dict
        List or dict with one color per segmentation.
    """
    if palette == 'husl':
        colors = husl_palette(len(segs), h=h, s=s, l=l)
    elif palette == 'hls':
        colors = hls_palette(len(segs), h=h, s=s, l=l)
    else:
        colors = color_palette(n_colors=len(segs), palette=palette)
    if isinstance(segs, dict):
        colors = {k: c for k, c in zip(segs.keys(), colors)}
    return colors


def stack_images(images, direction='horizontal', spacing=10):
    """Stack an iterable of images either veritcally or horizontally

    Parameters
    ----------
    images : list-like
        Iterable of Image.Image objects
    spacing : int, optional
        Spacing between images in pixels, by default 10
    direction : 'horizontal' or 'vertical', optional
        Direction of the grid of images, by default 'horizontal'

    Returns
    -------
    Image.Image
        Combined grid of images
    """
    if direction == 'horizontal':
        return _stack_horizontal(images, spacing=spacing)
    elif direction == 'vertical':
        return _stack_vertical(images, spacing=spacing)
    else:
        raise ValueError(
            'Direction must be either "horizontal" or "vertical".')


def _stack_horizontal(images, spacing=5):
    """Stack a set of equal height images with even spacing between them

    Parameters
    ----------
    images : list-like
        List of PIL.Image.Images with equal height
    spacing : int, optional
        Number of pixels between images horizontally, by default 5

    Returns
    -------
    PIL.Image.Image
        Grid of horizontally stacked images
    """
    images = [utils.convert_to_rgba(i) for i in images]
    h = np.max([i.height for i in images])
    ws = np.cumsum([0]+[i.width + spacing for i in images])
    w = ws[-1]-spacing

    img = Image.new('RGB', (w, h), (255, 255, 255, 255))
    for i, wd in zip(images, ws):
        img.paste(i, (wd, 0))
    return img


def _stack_vertical(images, spacing=10):
    """Stack a set images with even spacing between them

    Parameters
    ----------
    images : list-like
        List of PIL.Image.Images
    spacing : int, optional
        Number of pixels between images vertically, by default 10

    Returns
    -------
    PIL.Image.Image
        Grid of vertically stacked images aligned on left side.
    """
    images = [utils.convert_to_rgba(i) for i in images]
    w = np.max([i.width for i in images])
    hs = np.cumsum([0]+[i.height + spacing for i in images])
    h = hs[-1]-spacing
    img = Image.new('RGB', (w, h), (255, 255, 255, 255))
    for i, h in zip(images, hs):
        img.paste(i, (0, h))
    return img


def _composite_overlay_single(masks,
                              colors,
                              alpha=0.2,
                              imagery=None,
                              outline=False,
                              merge_outline=True,
                              overlap=True,
                              width=10,
                              side='out',
                              ):
    """Make a colored composite overlay from an iterable of 2d masks.

    Parameters
    ----------
    masks : list-like or dict
        Iterable of masked 2d images of the same size. If a dict, colors must be a dict as well.
    colors : list-like or dict
        Iterable of RGB colors of the same size as masks. If a dict, masks must also be a dict and colors
        must have all keys in masks.
    alpha : float, optional
        Alpha value for the overlay
    imagery : PIL.Image.Image or None, optional
        If an Image, applies the overlay to the image, by default None
    outline : bool, optional
        If True, produces an outline instead of a flat overay, by default False
    merge_outline : bool, optional
        If True, the merge outline applies to the segmentation as a whole and thus
        internal contacts are not outlined.
    overlap : bool, optional
        If False, segmentations later in the list will not overlap segmentations earlier on the list.
    width : int, optional
        If outline=True, sets the width of the outline, by default 10
    fill_holes : int, optional
        If above zero, sets the pixel size threshold for filling holes (see
        scipy.ndimage.fill_holes documentation), by default 0
    side : 'out' or 'in', optional
        If outline=True, selects if the outline is inside or outside the original segmentation mask, by default 'out'

    Returns
    -------
    PIL.Image.Image
        Composite overlay image, optionally overlaid over provided imagery.
    """

    if isinstance(masks, dict):
        mc_flat = [(masks[k], colors[k]) for k in masks.keys()]
        masks = [mc[0] for mc in mc_flat]
        colors = [mc[1] for mc in mc_flat]

    colors = utils.colors_to_uint8(colors)

    frames = []
    for mask, color in zip(masks, colors):
        dat = (np.array(mask) > 0)*255
        frame = Image.fromarray(np.squeeze(dat).T.astype('uint8'))
        if outline:
            frame = utils.binary_seg_outline(
                frame, width, side=side)
        frame = utils.colorize_bw(frame, new_color=color,
                                  new_alpha=alpha, background=0)
        frames.append(frame)

    composite_frames = Image.new('RGBA', size=frames[0].size)
    for f in frames:
        if overlap is False:
            _, _, _, a = np.array(composite_frames).T
            f = utils.mask_image(f, (a == 0).astype(int))
        composite_frames = Image.alpha_composite(composite_frames, f)

    if outline and merge_outline:
        all_mask = np.zeros(np.array(masks[0]).shape)
        for mask in masks:
            all_mask = all_mask + np.array(mask > 0).astype('int')
        all_mask = Image.fromarray(255*all_mask.astype('uint8').T)
        all_mask = utils.binary_seg_outline(
            all_mask, width, side=side)
        composite_frames = utils.mask_image(composite_frames, all_mask)

    if imagery is not None:
        imagery = utils.convert_to_rgba(imagery)
        return Image.alpha_composite(imagery, composite_frames)
    else:
        return composite_frames


def composite_overlay(segs,
                      colors=None,
                      alpha=0.2,
                      imagery=None,
                      outline=False,
                      merge_outline=True,
                      overlap=True,
                      width=10,
                      side='out',
                      dim=2,
                      palette=DEFAULT_PALETTE,
                      h=DEFAULT_H,
                      l=DEFAULT_L,
                      s=DEFAULT_S,
                      ):
    """Make a colored composite overlay for a 3d mask from an iterable of masks.

    Parameters
    ----------
    masks : list-like or dict
        Iterable of masked images of the same size. If a dict, colors must be a dict as well.
    colors : list-like, dict, or None
        Iterable of RGB colors of the same size as masks. If a dict, masks must also be a dict and colors
        must have all keys in masks. If None, uses `discrete_colors` to generate colors.
    alpha : float, optional
        Alpha value for the overlay
    imagery : PIL.Image.Image or None, optional
        If an Image, applies the overlay to the image, by default None
    outline : bool, optional
        If True, produces an outline instead of a flat overay, by default False
    merge_outline : bool, optional
        If True, the merge outline applies to the segmentation as a whole and thus
        internal contacts are not outlined.
    overlap : bool, optional
        If False, segmentations later in the list will not overlap segmentations earlier on the list.
    width : int, optional
        If outline=True, sets the width of the outline, by default 10
    side : 'out' or 'in', optional
        If outline=True, selects if the outline is inside or outside the original segmentation mask, by default 'out'
    dim : int, optional
        Determines axis over which slices are iterated if the data is 3 dimensional. Default is 2 (z-axis).

    Returns
    -------
    list or PIL.Image.Image
        Image or list of composite overlay images, optionally overlaid over provided imagery. List or single image
        is determined based on segmentation arrays being 2 or 3 dimensional.
    """
    if colors is None:
        colors = discrete_colors(segs, palette, h, l, s)

    n_dim = len(utils.get_first(segs).shape)
    if n_dim > 2:
        n_frames = utils.get_first(segs).shape[dim]
    else:
        return _composite_overlay_single(segs,
                                         colors,
                                         alpha=alpha,
                                         imagery=imagery,
                                         outline=outline,
                                         merge_outline=merge_outline,
                                         overlap=overlap,
                                         width=width,
                                         side=side,
                                         )

    overlay_images = []
    for ii in range(n_frames):
        frame_masks, frame_imagery = utils.flatten_data(
            ii, segs, imagery, dim)
        img = _composite_overlay_single(frame_masks,
                                        colors,
                                        alpha=alpha,
                                        imagery=frame_imagery,
                                        outline=outline,
                                        merge_outline=merge_outline,
                                        overlap=overlap,
                                        width=width,
                                        side=side,
                                        )
        overlay_images.append(img)
    return overlay_images
