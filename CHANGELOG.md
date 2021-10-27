# Changelog

This project attempts to follow [Semantic Versioning](https://semver.org) and uses [Keep-a-Changelog formatting](https://keepachangelog.com/en/1.0.0/).
But I make mistakes sometimes.

<!-- ## Unreleased -->
### Added

### Fixed

### Changed

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
