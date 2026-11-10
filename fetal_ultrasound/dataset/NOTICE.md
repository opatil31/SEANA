# FETAL_PLANES_DB v1 — native-region experiment inputs

Source: Burgos-Artizzu, X. P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B.,
Bonet-Carne, E., Eixarch, E., Crispi, F. and Gratacós, E. (2020).
FETAL_PLANES_DB: Common maternal-fetal ultrasound images, version 1.0.
https://doi.org/10.5281/zenodo.3904280

The upstream dataset is licensed under **Creative Commons Attribution 4.0
International (CC BY 4.0)**. The license is included as `LICENSE.txt`.
https://creativecommons.org/licenses/by/4.0/
The repository's MIT code license does not replace the dataset's license.

Please cite: Burgos-Artizzu et al., Evaluation of deep convolutional neural
networks for automatic classification of common maternal fetal ultrasound
planes. Scientific Reports 10, 10200 (2020).
https://doi.org/10.1038/s41598-020-67076-5

This release contains the prepared **192×192 native-resolution grayscale regions**
used by SEANA's fetal experiments, for all 12,400 source images. It is a prepared
subset of each frame, not a copy of the full-frame PNG collection. The preparation
matches the captured loader: convert to grayscale, find the bounding box of
pixels above intensity 4, reflect-pad the bottom/right if needed, and take a
centered 192×192 crop without interpolation. Cropped pixels are stored losslessly
as uint8 in `images.npy`; the loader divides by 255 and repeats the grayscale
channel three times. SSL views and probe inputs are cropped to 128×128 at runtime.
No denoised estimates, synthetic noise or trained weights are included.

The original metadata CSV and upstream README are retained. Added material consists
of a manifest, source-image and prepared-crop hashes, crop geometry, and explicit
patient partitions. The official split contains 7,129 training images and 5,271
test images from 896 patients each. The separate `official_validation` recipe
reserves 90 patients from the official training pool using a fixed seed; it is
a release addition, not an official or recovered historical validation split.

Device names are preserved from the source metadata: **Voluson S10**, Voluson E6,
Aloka and Other. Patient identifiers are the original public dataset identifiers.
