# Third-party components

The original SEANA code is MIT licensed. The following components retain separate
terms. This repository has not yet been published.

The `seana-ssl` library wheel contains only the original SEANA core and its
practitioner API, under MIT. Comparison implementations, datasets and pretrained
weights are excluded from the library distributions. The following comparison
and dataset notices apply to their files in the full research repository.

## LeJEPA / SIGReg

`cifar10/lejepa.py` is adapted from the project's microscopy implementation,
which identifies its Epps–Pulley and slicing routines as ports of
[galilai-group/lejepa](https://github.com/galilai-group/lejepa), by Randall
Balestriero and Yann LeCun. The upstream code is licensed under
[CC BY-NC 4.0](https://github.com/galilai-group/lejepa/blob/main/LICENSE).
The full license is retained in [third_party/lejepa-LICENSE.txt](third_party/lejepa-LICENSE.txt).
It is excluded from the MIT grant for original SEANA code.

The release adaptation runs on one process, uses two views at the selected dataset's size, and
exposes the upstream `(1-lambda)*invariance + lambda*SIGReg` weighting and the
captured project's `invariance + lambda*SIGReg` weighting as separate settings.
The upstream weighting is the default. Source reference:
[minimal example](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md).

## Next-Scale Prediction (NSP)

`denoisers/nsp.py` is the project's adaptation of
[XLearning-SCU/2026-CVPR-NSP](https://github.com/XLearning-SCU/2026-CVPR-NSP),
by Yiwen Shan, Haiyu Zhao, Peng Hu, Xi Peng and Yuanbiao Gou.
Its DBSNl architecture also references Wu et al., *Unpaired Learning of Deep
Image Denoising* (ECCV 2020).

No license file was present in the inspected NSP repository on 2026-11-06.
This local preparation preserves the user's existing adaptation; it does not
assert redistribution permission. **Resolve upstream redistribution terms before
publishing this file.** The root MIT license does not cover it. Adaptations
include `tds=4` for 32x32 images, a shorter fit schedule, functional masked
convolution, vectorized pair sampling and averaged sub-image inference.
The new ImageNet-100 recipe uses 160x160 fitting patches and tds=5.
The BBBC048 and fetal ultrasound recipes retain 64x64 fitting patches and tds=4.

## Neighbor2Neighbor (N2N)

`denoisers/n2n.py` and `denoisers/_n2n_unet.py` incorporate the sampler, objective
and U-Net from Tao Huang, Songjiang Li, Xu Jia, Huchuan Lu and Jianzhuang Liu,
*Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images* (CVPR 2021),
at [upstream commit e66389e](https://github.com/TaoHuang2018/Neighbor2Neighbor/tree/e66389ebef1e64d306d8fcb95f096cf427243452).
Copyright (c) 2021, Tao Huang. These components retain their **BSD-3-Clause**
license, reproduced in [third_party/neighbor2neighbor-LICENSE.txt](third_party/neighbor2neighbor-LICENSE.txt).
The original SEANA MIT grant does not replace that license.

[Source references and integration changes](denoisers/n2n_upstream.json) record
small-image reflection padding, CPU/CUDA sampling support, vectorized subimage
extraction, modern PyTorch compatibility and update-based fitting schedules.
[protocol.md](docs/protocol.md) specifies the objective and dataset recipes.
Captured historical N2N results used the earlier custom SmallUNet/objective and
are not measurements of this official-method implementation.

## Dependencies, data and pretrained weights

PyTorch and DeepInverse are the library's required dependencies.
The research workflows also use torchvision, Lightly, Hugging Face Datasets and tifffile.
Their source code is not bundled into this repository. Model heads and standard
losses use Lightly. The SEANA experiment uses DeepInverse's pretrained DRUNet;
its weights are obtained separately and retain their applicable terms.

CIFAR-10 data are obtained through torchvision or supplied by the user and are
not distributed here. ImageNet-100 is obtained separately through
[clane9/imagenet-100](https://huggingface.co/datasets/clane9/imagenet-100), whose
dataset card provides its terms. ImageNet images are not redistributed.
BBBC048's original pre-merged images and ground-truth list are included in the
single-file [dataset archive](fluorescence_imaging_bbbc048/dataset/bbbc048_merged_v1.zip).
The [dataset page](https://bbbc.broadinstitute.org/BBBC048) identifies Andrew Filby
as the licensor under **CC BY-NC-SA 3.0**. The archive retains that license and
attribution to the Flow Cytometry Core Facility, Newcastle University, with
citations to Eulenberg et al. (2017) and Ljosa et al. (2012). See
[the dataset notice](fluorescence_imaging_bbbc048/dataset/NOTICE.md) and
[full license](fluorescence_imaging_bbbc048/dataset/LICENSE.txt).
BBBC048 images are redistributed without pixel changes; SEANA adds the manifest,
hashes and reproducible split assignments.

FETAL_PLANES_DB's prepared native-resolution regions are included as the single
[fetal dataset file](fetal_ultrasound/dataset/fetal_planes_native192_v1.zip).
[Burgos-Artizzu et al.'s dataset v1](https://doi.org/10.5281/zenodo.3904280) is
licensed under **CC BY 4.0**. The file preserves the original metadata and adds
192x192 grayscale crops, provenance hashes and patient partitions. Cropping
matches the experiment loader without interpolation; the full source frames
are not bundled. See [attribution and modification notice](fetal_ultrasound/dataset/NOTICE.md)
and [full license](fetal_ultrasound/dataset/LICENSE.txt). The GitHub dataset file
uses Git LFS and is excluded from Python package distributions.

Experiment CSVs are the author's measurements.
The repository's MIT license does not grant rights to
datasets or pretrained weights.
