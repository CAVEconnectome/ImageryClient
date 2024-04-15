---
title: Getting Started
---

!!! important
    If using imageryclient on a CAVE-hosted dataset, we recommend installing [CAVEclient](https://caveclient.readthedocs.io/) for easier access to data. If so, please see the [CAVEclient documentation](caveconnectome.github.io/CAVEclient/) for more information.

We make use of [Numpy arrays](https://numpy.org/doc/stable/) and [Pillow Images](https://pillow.readthedocs.io/) to represent data.
Both are extremely rich tools, and to learn more about them please see the appropriate documentation for information about saving data to image files and more. 

## Installation

ImageryClient can be installed with pip:

`pip install imageryclient`

While not required, if you are using a CAVE-hosted dataset, installing [CAVEclient](https://caveclient.readthedocs.io/) will make your life much easier.

## Troubleshooting

If you have installation issues due to CloudVolume, which can have a complex set of requirements, we recommend looking at its github [issues page](https://github.com/seung-lab/cloud-volume/issues) for help.


## Basic example

A small example that uses all of the major components of ImageryClient: Downloading aligned images and segmentation, specifying specific segmentations to visualize, and generating an image overlay.
This uses the pubically available [Kasthuri et al. 2014 dataset](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.000000000000001e-9%2C%22m%22%5D%2C%22y%22:%5B6.000000000000001e-9%2C%22m%22%5D%2C%22z%22:%5B3.0000000000000004e-8%2C%22m%22%5D%7D%2C%22position%22:%5B5523.99072265625%2C8538.9384765625%2C1198.0423583984375%5D%2C%22projectionOrientation%22:%5B-0.0040475670248270035%2C-0.9566215872764587%2C-0.22688281536102295%2C-0.18271005153656006%5D%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/kasthuri2011/image%22%2C%22tab%22:%22source%22%2C%22name%22:%22original-image%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/kasthuri2011/image_color_corrected%22%2C%22tab%22:%22source%22%2C%22name%22:%22corrected-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://neuroglancer-public-data/kasthuri2011/ground_truth%22%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:0.63%2C%22notSelectedAlpha%22:0.14%2C%22segments%22:%5B%223208%22%2C%224901%22%2C%2213%22%2C%224965%22%2C%224651%22%2C%222282%22%2C%223189%22%2C%223758%22%2C%2215%22%2C%224027%22%2C%223228%22%2C%22444%22%2C%223207%22%2C%223224%22%2C%223710%22%5D%2C%22name%22:%22ground_truth%22%7D%5D%2C%22layout%22:%224panel%22%7D).

```python
import imageryclient as ic

img_src = 'precomputed://gs://neuroglancer-public-data/kasthuri2011/image_color_corrected'
seg_src = 'precomputed://gs://neuroglancer-public-data/kasthuri2011/ground_truth'

img_client = ic.ImageryClient(image_source=img_src, segmentation_source=seg_src)

bounds = [
    [5119, 8477, 1201],
    [5519, 8877, 1202]
]
root_ids = [2282, 4845]

image, segs = img_client.image_and_segmentation_cutout(bounds,
                                                       split_segmentations=True,
                                                       root_ids=root_ids)


ic.composite_overlay(segs, imagery=image)
```

![Expected imagery overlay from the code above](images/seg_overlay_0.png)

