import numpy as np
import imageio
import cloudvolume as cv
from scipy import ndimage
from functools import partial
import datetime
from . import utils


def bounds_from_center(ctr, width=1, height=1, depth=1):
    """Generate bounds from a center point and dimensions for each direction

    Parameters
    ----------
    ctr : array-like
        x,y,z coordinates of the center of the bounds in voxel dimensions.
    width: int, optional
        Width of the box in the x direction in.
        Default is 1.
    height: int, optional
        Height of the box in the y direction.
        Default is 1.
    depth: int, optional
        Depth of the box in the z direction.
        Default is 1.

    Returns
    -------
    array
        2x3 array of lower and upper bounds (in that order).
    """
    xl = ctr - np.array([width // 2, height // 2, depth // 2])
    xh = xl + np.array([width, height, depth])
    return np.array([xl, xh])



def save_image_slices(
    filename_prefix,
    filename_suffix,
    img,
    slice_axis,
    image_type,
    verbose=False,
    color=None,
    **kwargs,
):
    """Helper function for generic image saving"""
    if image_type == "imagery":
        to_pil = utils._grayscale_to_pil
    elif image_type == "mask":
        to_pil = partial(utils._binary_mask_to_transparent_pil, color=color)

    imgs = np.split(img, img.shape[slice_axis], axis=slice_axis)
    if len(imgs) == 1:
        fname = f"{filename_prefix}_{filename_suffix}.png"
        imageio.imwrite(fname, to_pil(imgs[0].squeeze()), **kwargs)
        if verbose:
            print(f"Saved {fname}...")
    else:
        for ii, img_slice in enumerate(imgs):
            fname = f"{filename_prefix}_slice_{ii}_{filename_suffix}.png"
            imageio.imwrite(fname, to_pil(img_slice.squeeze()), **kwargs)
            if verbose:
                print(f"Saved {fname}...")
    return


class ImageryClient(object):
    """Tool to help download imagery and segmentation data.

    Can either take explicit cloudvolume paths for imagery and segmentation or use the Info Service to look up the right paths.

    Parameters
    ----------
    image_source : str, optional
        CloudVolume path to an imagery source, by default None
    segmentation_source : str, optional
        CloudVolume path to a segmentation source, by default None
    datastack_name : str, optional
        Datastack name to lookup information for in the InfoService, by default None
    server_address : str, optional
        Address of an InfoService host, by default None. If none, uses defaults in
        the CAVEclient.
    base_resolution : array-like or 'image' or 'segmentation', optional
        Sets the voxel resolution that bounds will be entered in, by default 'image'.
        Literal resolutions will be followed, while "image" or "segmentation" use the
        mip 0 values of the associated cloudvolumes.
    table_name : str, optional
        Name of the chunkedgraph table (if used), by default None
    image_mip : int, optional
        Default mip level to use for imagery lookups, by default 0. Note that the same mip
        level for imagery and segmentation can correspond to different voxel resolutions.
    segmentation_mip : int, optional
        Default mip level to use for segmentation lookups, by default 0.
    segmentation : bool, optional
        If False, no segmentation cloudvolume is initialized. By default True
    imagery : bool, optional
        If False, no imagery cloudvolume is initialized. By default True
    framework_client : caveclient.CAVEclient, optional
        A pre-initialized Framework client to be used instead of initializing a new one.
    auth_token : str or None, optional
        Auth token to use for cloudvolume. If None, uses the default values from the CAVEclient. Default is None.
    timestamp : datetime.datetime or None,
        Fixed timestamp to use for segmentation lookups. If None, defaults to the present time
        when each function is run. Default is None.
    """

    def __init__(
        self,
        client=None,
        resolution=None,
        segmentation=True,
        imagery=True,
        image_source=None,
        segmentation_source=None,
        image_mip=None,
        segmentation_mip=None,
        auth_token=None,
        timestamp=None,
    ):

        self._image_source = image_source
        self._segmentation_source = segmentation_source

        self._auth_token = None
        if auth_token is not None:
            self._auth_token = auth_token

        if timestamp == "now":
            self._timestamp = datetime.datetime.now().timestamp()
        elif timestamp is None:
            self._timestamp = None
        else:
            self._timestamp = timestamp

        self._use_segmentation = segmentation
        self._use_imagery = imagery
        self._img_cv = None
        self._seg_cv = None

        self._resolution = resolution
        if client is not None:
            self._configure_from_client(client)
        self._configure_mip_levels(image_mip, segmentation_mip)
        self._configure_resolution(self._resolution)

    def _configure_from_client(self, client):
        if self._auth_token is None:
            self._auth_token = client.auth.token
        if self._image_source is None and self._use_imagery:
            self._image_source = client.info.image_source()
        if self._segmentation_source is None and self._use_segmentation:
            self._segmentation_source = client.info.segmentation_source()
        if self._resolution is None:
            self._resolution = client.info.viewer_resolution()

    def _configure_mip_levels(self, image_mip, segmentation_mip):
        if image_mip is None:
            self._base_imagery_mip = utils._get_lowest_nonplaceholder(self.image_cv)
        else:
            self._base_imagery_mip = image_mip
        if segmentation_mip is None:
            self._base_segmentation_mip = utils._get_lowest_nonplaceholder(self.segmentation_cv)        
        else:
            self._base_segmentation_mip = segmentation_mip

    def _configure_resolution(self, resolution):
        if resolution is None:
            resolution = "image"
        
        if isinstance(resolution, str):
            if resolution in ["image", "segmentation"]:
                if resolution == "image":
                    if self._use_imagery is None:
                        raise ValueError(
                            "Cannot set resolution from imagery if not being used"
                        )
                    self._resolution = np.array(self.image_cv.mip_resolution(self._base_imagery_mip))
                elif resolution == "segmentation":
                    if self._use_segmentation is None:
                        raise ValueError(
                            "Cannot set resolution from segmentation if not being used"
                        )
                    self._resolution = np.array(self.segmentation_cv.mip_resolution(self._base_segmentation_mip))
            else:
                raise ValueError(
                    'Base resolution must be set by the client, array-like, "image" or "segmentation"'
                )
        else:
            self._resolution = np.array(resolution)


    @property
    def token(self):
        return self._auth_token

    @property
    def timestamp(self):
        if self._timestamp is None:
            return datetime.datetime.now().timestamp()
        return self._timestamp

    @property
    def resolution(self):
        """The voxel resolution assumed when locations are used for the client.

        Returns
        -------
        list
            X, y, and z voxel resolution.
        """
        return self._resolution

    @property
    def image_source(self):
        """Image cloudpath"""
        return self._image_source

    @property
    def segmentation_source(self):
        """Segmentation cloudpath"""
        return self._segmentation_source

    @property
    def image_cv(self):
        """Imagery CloudVolume"""
        if self._use_imagery is False:
            return None

        if self._img_cv is None:
            self._img_cv = cv.CloudVolume(
                self.image_source,
                bounded=False,
                fill_missing=True,
                progress=False,
                use_https=True,
                secrets=self.token,
            )
        return self._img_cv

    @property
    def segmentation_cv(self):
        """Segmentation CloudVolume object"""
        if self._use_segmentation is False:
            return None

        elif self._seg_cv is None:
            self._seg_cv = cv.CloudVolume(
                self.segmentation_source,
                use_https=True,
                fill_missing=True,
                bounded=False,
                progress=False,
                secrets=self.token,
            )
        return self._seg_cv

    def _bounds_to_slices(self, bounds):
        return cv.Bbox(bounds[0], bounds[1])

    def _compute_dimensions_from_image_size(
        self,
        image_size,
        mip_resolution,
        resolution,
    ):
        if len(image_size) == 3:
            width, height, depth = image_size
            return [
                int(width) * mip_resolution[0] / resolution[0],
                int(height) * mip_resolution[1] / resolution[1],
                int(depth) * mip_resolution[2] / resolution[2],
            ]
        elif len(image_size) == 2:
            width, height = image_size
            return [
                int(width) * mip_resolution[0] / resolution[0],
                int(height) * mip_resolution[1] / resolution[1],
                1,
            ]

    def image_bbox_size_from_dimensions(self, image_size, mip=None, resolution=None):
        """Get the bbox_size equivalent for an imagery cutout with specified pixel dimensions

        Parameters
        ----------
            image_size: list-like
                Image size in pixels (2-element) or voxels (3-element)
            mip: int or None, optional
                Mip for which the image would be computed. Defaults to None, which uses the client default.
            resolution: list-like or None, optional.
                Resolution to use for the bbox_size. Defaults to None, or the client defauls.

        Returns
        -------
            tuple:
                Argument for bbox_size that would give the desired pixel dimensions.
        """
        if mip is None:
            mip = self._base_imagery_mip
        if resolution is None:
            resolution = self._resolution
        return self._compute_dimensions_from_image_size(
            image_size, self.image_cv.mip_resolution(mip), resolution
        )

    def segmentation_bbox_size_from_dimensions(
        self, image_size, mip=None, resolution=None
    ):
        """Get the bbox_size equivalent for an segmentation cutout with specified pixel dimensions

        Parameters
        ----------
            image_size: list-like
                Image size in pixels (2-element) or voxels (3-element)
            mip: int or None, optional
                Mip for which the image would be computed. Defaults to None, which uses the client default.
            resolution: list-like or None, optional.
                Resolution to use for the bbox_size. Defaults to None, or the client defauls.

        Returns
        -------
            tuple:
                Argument for bbox_size that would give the desired pixel dimensions.
        """

        if mip is None:
            mip = self._base_segmentation_mip
        if resolution is None:
            resolution = self._resolution
        return self._compute_dimensions_from_image_size(
            image_size, self.segmentation_cv.mip_resolution(mip), resolution
        )

    def _compute_bounds(self, bounds, bbox_size):
        if bbox_size is not None:
            if len(bbox_size) == 3:
                width, height, depth = bbox_size
            elif len(bbox_size) == 2:
                width, height = bbox_size
                depth = 1
            bounds = bounds_from_center(
                np.atleast_2d(bounds)[0], width=width, height=height, depth=depth
            )
        return self._bounds_to_slices(bounds)

    def image_cutout(
        self,
        bounds,
        bbox_size=None,
        image_size=None,
        mip=None,
        resolution=None,
        scale_to_bounds=None,
    ):
        """Get an image cutout for a certain location or set of bounds and a mip level.

        Parameters
        ----------
        bounds : array
            Either a 2x3 array of a lower bound and upper bound point to bound the cutout in units of voxels in a resolution set by
            the base_resolution parameter. Alternatively, if bbox_size or image_size is set, bounds should be a 3-element array specifying the center of the
            field of view.
        bbox_size : array or None, optional
            If not None, bbox_size is a 3 element array of ints giving the dimensions. In this case, bounds is treated
            as the center.
        image_size : array or None, optional
            If not None, indicates the size of the image desired in pixels. Cannot be set with bbox_size, since it has potentially conficting information.
        mip : int, optional
            Mip level of imagery to get if something other than the default is wanted, by default None.
        resolution : list-like, optional
            Voxel resolution used when specifying bounds bounds and bbox_size (but not image_size). If none, defaults to client default.
        scale_to_bounds : bool, optional.
            If True, rescales image to the same size as the bounds. Default is None, which rescales if mip is not set but otherwise does not.

        Returns
        -------
            cloudvolume.VolumeCutout
            An n-d image of the image requested with image intensity values as the elements.
        """
        if self.image_cv is None:
            return np.array([])

        if resolution is None:
            resolution = self.resolution
        if mip is None:
            mip = self._base_imagery_mip
        else:
            if scale_to_bounds is None:
                scale_to_bounds = False

        if image_size is not None:
            if bbox_size is not None:
                raise ValueError("Cannot set both image size and bbox size!")
            bbox_size = self._compute_dimensions_from_image_size(
                image_size, self.image_cv.mip_resolution(mip), resolution
            )

        bbox = self._compute_bounds(bounds, bbox_size)
        img = np.array(
            np.squeeze(
                self.image_cv.download(bbox, coord_resolution=resolution, mip=mip)
            )
        )
        if scale_to_bounds:
            return utils.rescale_to_bounds(
                img,
                self._compute_bounds(bounds, bbox_size),  # downloading changes the bbox
            )
        else:
            return img

    def segmentation_cutout(
        self,
        bounds,
        root_ids="all",
        bbox_size=None,
        image_size=None,
        mip=None,
        resolution=None,
        timestamp=None,
        scale_to_bounds=None,
    ):
        """Get a cutout of the segmentation imagery for some or all root ids between set bounds.
        Note that if all root ids are requested in a large region, it could take a long time to query
        all supervoxels.

        Parameters
        ----------
        bounds : 2x3 list of ints
            A list of the lower and upper bound point for the cutout. The units are voxels in the resolution set by the
            base_resolution parameter.
        root_ids : list, None, or 'all', optional
            If a list, only compute the voxels for a specified set of root ids. If None, default to the supervoxel ids. If 'all',
            find all root ids corresponding to the supervoxels in the cutout and get all of them. None, by default 'all'
        bbox_size : array or None, optional
            If not None, bbox_size is a 3 element array of ints giving the dimensions. In this case, bounds is treated
            as the center.
        image_size : array or None, optional
            If not None, indicates the size of the image desired in pixels. Cannot be set with bbox_size, since it has potentially conficting information.
        mip : int, optional
            Mip level of the segmentation if something other than the defualt is wanted, by default None
        resolution : list-like, optional
            Voxel resolution used when specifying bounds bounds and bbox_size (but not image_size). If none, defaults to client default.
        timestamp : datetime or None, optional
            Timestamp to use for dynamic segmentation data
        scale_to_bounds : bool or None, optional
            If True, rescales image to the same size as the bounds. Default is None, which rescales if mip is not set but otherwise does not.

        Returns
        -------
        numpy.ndarray
            Array whose elements correspond to the root id (or, if root_ids=None, the supervoxel id) at each voxel.
        """
        if self.segmentation_cv is None:
            return np.array([])
        if mip is None:
            mip = self._base_segmentation_mip
        else:
            if scale_to_bounds is None:
                scale_to_bounds = False
        if resolution is None:
            resolution = self.resolution
        if timestamp is None:
            timestamp = self._timestamp

        if image_size is not None:
            if bbox_size is not None:
                raise ValueError("Cannot set both image size and bbox size!")
            bbox_size = self._compute_dimensions_from_image_size(
                image_size, self.segmentation_cv.mip_resolution(mip), resolution
            )
        bbox = self._compute_bounds(bounds, bbox_size)

        if isinstance(root_ids, str):
            if root_ids == "all":
                seg = np.array(
                    np.squeeze(
                        self.segmentation_cv.download(
                            bbox,
                            agglomerate=True,
                            mip=mip,
                            coord_resolution=resolution,
                            timestamp=timestamp,
                        )
                    )
                )
            else:
                raise ValueError('root_ids must be None, list, or "all')
        else:
            seg = np.array(
                np.squeeze(
                    self.segmentation_cv.download(
                        bbox,
                        segids=root_ids,
                        mip=mip,
                        coord_resolution=resolution,
                        timestamp=timestamp,
                    )
                )
            )

        if scale_to_bounds:
            return utils.rescale_to_bounds(
                seg,
                self._compute_bounds(bounds, bbox_size),  # downloading changes the bbox
            )
        else:
            return seg

    def split_segmentation_cutout(
        self,
        bounds,
        root_ids="all",
        include_null_root=False,
        bbox_size=None,
        image_size=None,
        mip=None,
        resolution=None,
        timestamp=None,
        scale_to_bounds=None,
    ):
        """Generate segmentation cutouts with a single binary mask for each root id, organized as a dict with keys as root ids and masks as values.

        Parameters
        ----------
        bounds : 2x3 list of ints
            A list of the lower and upper bound point for the cutout. The units are voxels in the resolution set by the
            base_resolution parameter.
        root_ids : list, None, or 'all', optional
            If a list, only compute the voxels for a specified set of root ids. If None, default to the supervoxel ids. If 'all',
            find all root ids corresponding to the supervoxels in the cutout and get all of them. None, by default 'all'
        include_null_root : bool, optional
            If True, includes root id of 0, which is usually reserved for a null segmentation value. Default is False.
        bbox_size : array or None, optional
            If not None, bbox_size is a 3 element array of ints giving the dimensions. In this case, bounds is treated
            as the center.
        image_size : array or None, optional
            If not None, indicates the size of the image desired in pixels. Cannot be set with bbox_size, since it has potentially conficting information.
        mip : int, optional
            Mip level of the segmentation if something other than the default is wanted, by default None
        resolution : list-like, optional
            Voxel resolution used when specifying bounds bounds and bbox_size (but not image_size). If none, defaults to client default.
        timestamp : datetime or None, optional
            Timestamp to use for dynamic segmentation data
        scale_to_bounds : bool or None, optional
            If True, rescales image to the same size as the bounds. Default is None, which rescales if mip is not set but otherwise does not.

        Returns
        -------
        dict
            Dict whose keys are root ids and whose values are the binary mask for that root id, with a 1 where the object contains the voxel.
        """
        seg_img = self.segmentation_cutout(
            bounds=bounds,
            root_ids=root_ids,
            mip=mip,
            bbox_size=bbox_size,
            image_size=image_size,
            resolution=resolution,
            timestamp=timestamp,
            scale_to_bounds=scale_to_bounds,
        )
        return utils.segmentation_masks(seg_img, include_null_root)

    def image_and_segmentation_cutout(
        self,
        bounds,
        image_mip=None,
        segmentation_mip=None,
        root_ids="all",
        resize=True,
        split_segmentations=False,
        include_null_root=False,
        bbox_size=None,
        resolution=None,
        timestamp=None,
        scale_to_bounds=None,
    ):

        """Download aligned and scaled imagery and segmentation data at a given resolution.

        Parameters
        ----------
        bounds : 2x3 list of ints
            A list of the lower and upper bound point for the cutout. The units are voxels in the resolution set by the
            base_resolution parameter.
        image_mip : int, optional
            Mip level of the imagery if something other than the default is wanted, by default None
        segmentation_mip : int, optional
            Mip level of the segmentation if something other than the default is wanted, by default None
        root_ids : list, None, or 'all', optional
            If a list, the segmentation cutout only includes voxels for a specified set of root ids.
            If None, default to the supervoxel ids. If 'all', finds all root ids corresponding to the supervoxels
            in the cutout and get all of them. By default 'all'.
        resize : bool, optional
            If True, upscale the lower resolution cutout to the same resolution of the higher one (either imagery or segmentation).
        split_segmentations : bool, optional
            If True, the segmentation is returned as a dict of masks (using split_segmentation_cutout), and if False returned as
            an array with root_ids (using segmentation_cutout), by default False
        include_null_root : bool, optional
            If True, includes root id of 0, which is usually reserved for a null segmentation value. Default is False.
        bbox_size : array or None, optional
            If not None, bbox_size is a 3 element array of ints giving the dimensions of the cutout. In this case, bounds is treated
            as the center.
        resolution : list-like, optional
            Voxel resolution used when specifying bounds bounds and bbox_size (but not image_size). If none, defaults to client default.
        timestamp : datetime or None, optional
            Timestamp to use for dynamic segmentation data
        scale_to_bounds : bool or None, optional
            If True, rescales image to the same size as the bounds. Default is None, which rescales if mip is not set but otherwise does not.


        Returns
        -------
        cloudvolume.VolumeCutout
            Imagery volume cutout

        numpy.ndarray or dict
            Segmentation volume cutout as either an ndarray or dict of masks depending on the split_segmentations flag.
        """
        if image_mip is None:
            image_mip = self._base_imagery_mip
        if segmentation_mip is None:
            segmentation_mip = self._base_segmentation_mip
        if resolution is None:
            resolution = self._resolution

        img_resolution = self.image_cv.mip_resolution(image_mip)
        seg_resolution = self.segmentation_cv.mip_resolution(segmentation_mip)

        if np.all(img_resolution == seg_resolution):
            zoom_to = None
            scale_to_bounds_seg = scale_to_bounds
            scale_to_bounds_img = scale_to_bounds
        elif np.all(img_resolution >= seg_resolution):
            zoom_to = "segmentation"
            scale_to_bounds_seg = scale_to_bounds
            scale_to_bounds_img = False
        else:
            zoom_to = "image"
            scale_to_bounds_seg = False
            scale_to_bounds_img = scale_to_bounds

        img = self.image_cutout(
            bounds=bounds,
            mip=image_mip,
            bbox_size=bbox_size,
            resolution=resolution,
            scale_to_bounds=scale_to_bounds_img,
        )
        img_shape = img.shape

        if split_segmentations is False:
            seg = self.segmentation_cutout(
                bounds,
                root_ids=root_ids,
                mip=segmentation_mip,
                bbox_size=bbox_size,
                resolution=resolution,
                timestamp=timestamp,
                scale_to_bounds=scale_to_bounds_seg,
            )
            seg_shape = seg.shape
        else:
            seg = self.split_segmentation_cutout(
                bounds,
                root_ids=root_ids,
                mip=segmentation_mip,
                include_null_root=include_null_root,
                bbox_size=bbox_size,
                resolution=resolution,
                timestamp=timestamp,
                scale_to_bounds=scale_to_bounds_seg,
            )
            if len(seg) > 0:
                seg_shape = seg[list(seg.keys())[0]].shape
            else:
                seg_shape = 1

        if resize is False:
            pass
        elif zoom_to == "segmentation":
            zoom_scale = np.array(seg_shape) / np.array(img_shape)
            img = ndimage.zoom(img, zoom_scale, mode="nearest", order=0)
        elif zoom_to == "image":
            zoom_scale = np.array(img_shape) / np.array(seg_shape)
            if split_segmentations is False:
                seg = ndimage.zoom(seg, zoom_scale, mode="nearest", order=0)
            else:
                for root_id, seg_cutout in seg.items():
                    seg[root_id] = ndimage.zoom(
                        seg_cutout, zoom_scale, mode="nearest", order=0
                    )
        return img, seg

    def save_imagery(
        self,
        filename_prefix,
        bounds=None,
        mip=None,
        precomputed_image=None,
        slice_axis=2,
        bbox_size=None,
        image_size=None,
        resolution=None,
        scale_to_bounds=None,
        verbose=False,
        **kwargs,
    ):
        """Save queried or precomputed imagery to png files.

        Parameters
        ----------
        filename_prefix : str
            Prefix for the imagery filename. The full filename will be {filename_prefix}_imagery.png
        bounds : 2x3 list of ints, optional
            A list of the lower and upper bound point for the cutout. The units are voxels in the resolution set by the
            base_resolution parameter. Only used if a precomputed image is not passed. By default, None.
        mip : int, optional
            Only used if a precomputed image is not passed. Mip level of imagery to get if something other than the default
            is wanted, by default None
        precomputed_image : cloudvolume.VolumeCutout, optional
            Already downloaded VolumeCutout data to save explicitly. If called this way, the bounds and mip arguments will not apply.
            If a precomputed image is not provided, bounds must be specified to download the cutout data. By default None
        slice_axis : int, optional
            If the image data is truly 3 dimensional, determines which axis to use to save serial images, by default 2 (i.e. z-axis)
        bbox_size : array or None, optional
            If not None, bbox_size is a 3 element array of ints giving the dimensions of the cutout. In this case, bounds is treated
            as the center.
        image_size : array or None, optional
            If not None, indicates the size of the image desired in pixels. Cannot be set with bbox_size, since it has potentially conficting information.
        resolution : list-like, optional
            Voxel resolution used when specifying bounds bounds and bbox_size (but not image_size). If none, defaults to client default.
        scale_to_bounds : bool or None, optional
            If True, rescales image to the same size as the bounds. Default is None, which rescales if mip is not set but otherwise does not.
        verbose : bool, optional
            If True, prints the progress, by default False
        """
        if precomputed_image is None:
            img = self.image_cutout(
                bounds,
                mip=mip,
                bbox_size=bbox_size,
                image_size=image_size,
                resolution=resolution,
                scale_to_bounds=scale_to_bounds,
            )
        else:
            img = precomputed_image
        save_image_slices(
            filename_prefix,
            "imagery",
            img,
            slice_axis,
            "imagery",
            verbose=verbose,
            **kwargs,
        )
        pass

    def save_segmentation_masks(
        self,
        filename_prefix,
        bounds=None,
        mip=None,
        root_ids="all",
        precomputed_masks=None,
        segmentation_colormap={},
        slice_axis=2,
        include_null_root=False,
        bbox_size=None,
        image_size=None,
        resolution=None,
        timestamp=None,
        scale_to_bounds=None,
        verbose=False,
        **kwargs,
    ):
        """Save queried or precomputed segmentation masks to png files. Additional kwargs are passed to imageio.imwrite.

        Parameters
        ----------
        filename_prefix : str
            Prefix for the segmentation filenames. The full filename will be either {filename_prefix}_root_id_{root_id}.png
            or {filename_prefix}_root_id_{root_id}_{i}.png, depending on if multiple slices of each root id are saved.
        bounds : 2x3 list of ints, optional
            A list of the lower and upper bound point for the cutout. The units are voxels in the resolution specified.
            Only used if a precomputed segmentation is not passed. By default, None.
        mip : int, optional
            Only used if a precomputed segmentation is not passed. Mip level of segmentation to get if something other than the default
            is wanted, by default None
        root_ids : list, None, or 'all', optional
            If a list, the segmentation cutout only includes voxels for a specified set of root ids.
            If None, default to the supervoxel ids. If 'all', finds all root ids corresponding to the supervoxels
            in the cutout and get all of them. By default 'all'.
        precomputed_masks : dict, optional
            Already downloaded dict of mask data to save explicitly. If called this way, the bounds and mip arguments will not apply.
            If precomputed_masks are not provided, bounds must be given to download cutout data. By default None
        segmentation_colormap : dict, optional
            A dict of root ids to an uint8 RGB color triplet (0-255) or RGBa quadrooplet to optionally color the mask png. Any root id not specified
            will be rendered in white. Color triplets default to full opacity. Default is an empty dictionary.
        slice_axis : int, optional
            If the image data is truly 3 dimensional, determines which axis to use to save serial images, by default 2 (i.e. z-axis)
        include_null_root : bool, optional
            If True, includes root id of 0, which is usually reserved for a null segmentation value. Default is False.
        bbox_size : array or None, optional
            If not None, bbox_size is a 3 element array of ints giving the dimensions of the cutout. In this case, bounds is treated
            as the center.
        image_size : array or None, optional
            If not None, indicates the size of the image desired in pixels. Cannot be set with bbox_size, since it has potentially conficting information.
        resolution : list-like, optional
            Voxel resolution used when specifying bounds bounds and bbox_size (but not image_size). If none, defaults to client default.
        timestamp : datetime or None, optional
            Timestamp to use for dynamic segmentation data
        scale_to_bounds : bool or None, optional
            If True, rescales image to the same size as the bounds. Default is None, which rescales if mip is not set but otherwise does not.
        """
        if precomputed_masks is None:
            seg_dict = self.split_segmentation_cutout(
                bounds=bounds,
                root_ids=root_ids,
                mip=mip,
                bbox_size=bbox_size,
                image_size=image_size,
                include_null_root=include_null_root,
                resolution=resolution,
                timestamp=timestamp,
                scale_to_bounds=scale_to_bounds,
            )
        else:
            seg_dict = precomputed_masks

        for root_id, seg_mask in seg_dict.items():
            suffix = f"root_id_{root_id}"
            save_image_slices(
                filename_prefix,
                suffix,
                seg_mask,
                slice_axis,
                "mask",
                color=segmentation_colormap.get(root_id, None),
                verbose=verbose,
                **kwargs,
            )
        pass

    def save_image_and_segmentation_masks(
        self,
        filename_prefix,
        bounds=None,
        image_mip=None,
        segmentation_mip=None,
        root_ids="all",
        include_null_root=False,
        segmentation_colormap={},
        resize=True,
        precomputed_data=None,
        slice_axis=2,
        bbox_size=None,
        resolution=None,
        timestamp=None,
        scale_to_bounds=None,
        **kwargs,
    ):
        """Save aligned and scaled imagery and segmentation mask cutouts as pngs. Kwargs are passed to imageio.imwrite.

        Parameters
        ----------
        filename_prefix : str
            Prefix for the segmentation filenames. The full filename will be either {filename_prefix}_root_id_{root_id}.png
            or {filename_prefix}_root_id_{root_id}_{i}.png, depending on if multiple slices of each root id are saved.
        bounds : 2x3 list of ints, optional
            A list of the lower and upper bound point for the cutout. The units are voxels in the resolution set by the
            base_resolution parameter. Only used if a precomputed data is not passed. By default, None.
        image_mip : int, optional
            Only used if a precomputed data is not passed. Mip level of imagery to get if something other than the default
            is wanted, by default None.
        segmentation_mip : int, optional
            Only used if precomputed data is not passed. Mip level of segmentation to get if something other than the default
            is wanted, by default None
        root_ids : list, None, or 'all', optional
            If a list, the segmentation cutout only includes voxels for a specified set of root ids.
            If None, default to the supervoxel ids. If 'all', finds all root ids corresponding to the supervoxels
            in the cutout and get all of them. By default 'all'.
        include_null_root : bool, optional
            If True, includes root id of 0, which is usually reserved for a null segmentation value. By default, False.
        segmentation_colormap : dict, optional
            A dict of root ids to an uint8 RGB color triplet (0-255) or RGBa quadrooplet to optionally color the mask png. Any root id not specified
            will be rendered in white. Color triplets default to full opacity. Default is an empty dictionary.
        resize : bool, optional
            If True, upscale the lower resolution cutout to the same resolution of the higher one (either imagery or segmentation).
        precomputed_data : tuple, optional
            Already computed tuple with imagery and segmentation mask data, in that order. If not provided, bounds must be given to download
            cutout data. By default, None.
        slice_axis : int, optional
            If the image data is truly 3 dimensional, determines which axis to use to save serial images, by default 2 (i.e. z-axis)
        bbox_size : array or None, optional
            If not None, bbox_size is a 3 element array of ints giving the dimensions of the cutout. In this case, bounds is treated
            as the center.
        resolution : list-like, optional
            Voxel resolution of the bounds provided. If not provided, uses the client defaults.
        timestamp : datetime, optional
            Timestamp to use for the segmentation. If not provided, defaults to the client defaults.
        scale_to_bounds : bool or None, optional
            If True, rescales image to the same size as the bounds. Default is None, which rescales if mip is not set but otherwise does not.

        """
        if precomputed_data is not None:
            img, seg_dict = precomputed_data
        else:
            img, seg_dict = self.image_and_segmentation_cutout(
                bounds=bounds,
                image_mip=image_mip,
                segmentation_mip=segmentation_mip,
                bbox_size=bbox_size,
                root_ids=root_ids,
                resize=resize,
                include_null_root=include_null_root,
                split_segmentations=True,
                resolution=resolution,
                timestamp=timestamp,
                scale_to_bounds=scale_to_bounds,
            )

        self.save_imagery(
            filename_prefix,
            precomputed_image=img,
            slice_axis=slice_axis,
            **kwargs,
        )
        self.save_segmentation_masks(
            filename_prefix,
            precomputed_masks=seg_dict,
            slice_axis=slice_axis,
            segmentation_colormap=segmentation_colormap,
            **kwargs,
        )
        pass
