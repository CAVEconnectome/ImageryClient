# Changelog

This project attempts to follow [Semantic Versioning](https://semver.org) and uses [Keep-a-Changelog formatting](https://keepachangelog.com/en/1.0.0/).
But I make mistakes sometimes.

## [1.0.1] - 2022-11-28
### Fixed

- By default, the client will adopt the lowest mip level (i.e. highest resolution data) that is not labeled as a placeholder in the cloudvolume info.

## [1.0.0] - 2022-11-27
### Added

- Richer integration of specified resolution, to take advantage of better Cloudvolume support for resolutions.
- Split apart the notion of bounding box dimensions from image dimensions. See bbox_size vs image_size in teh readme.
- Vastly improved configuration by a CAVEclient
### Fixed

- Order of arguments is more consistent across functions, and more sensible as well.
## [0.4.0] - 2021-10-27
### Added

- 'voxel_dimensions' arguments can be used to specify pixel dimensions instead of voxel dimensions. If used, the 'bounds' argument instead specifies the center

### Fixed

- Switched from annotationframeworkclient to caveclient
- No longer need caveclient (although it's still useful)
- Directly saving files is allowed

### Changed

- Changed behavior of bounds_from_center to use width/height/depth instead of radius-style values
- More direct use of newer cloudvolume features
- Cloudvolume version 3.1.0 or greater is now required

## [0.2.3] - 2020-09-24

### Fixed

- Fix install bug

## [0.2.2] — 2020-09-13

### Fixed

- Adding long description for pypi

## [0.2.1] — 2020-09-13

### Fixed

- Added seaborn to requirements

## [0.2.0] — 2020-09-13

### Added

- Color generation using the seaborn color palettes.
- Documentation updated.

### Changed

- Color specification is now optional in overlays and a default color picker is used.

## [0.1.0] — 2020-09-12

Initial release, for all intents and purposes.
