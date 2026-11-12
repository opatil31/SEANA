# BBBC048 demo in Google Colab

The root [SEANA_BBBC048_Demo.ipynb](../SEANA_BBBC048_Demo.ipynb) provides:

- Preserved LeJEPA final-head results read from checksum-verified captured CSVs,
  clearly labeled as historical evidence.
- Real BBBC048 cell examples and a VST fitted on training images without clean targets.
- The published `seana-ssl==0.1.3` package, with pretrained DRUNet as its default denoiser.
- An interactive comparison of acquired images, DRUNet alone, the learned-VST
  estimate and independently resampled views, plus a plot of the learned transform.
- A matched baseline/SEANA experiment with all five SSL models available.
  The results chart and printed summary show Top-1 accuracy and macro-F1.
- A compact downloadable record with metrics, memberships, settings, source and
  environment hashes, and the comparison figure.

## Open the notebook

1. Download the root notebook and open [Google Colab](https://colab.research.google.com/).
   Choose **File → Upload notebook** and upload the `.ipynb`.
2. Prepare the upload bundle from the repository root (standard-library Python only):

   ```bash
   python -m scripts.make_colab_bundle --output runs/SEANA_BBBC048_Colab.zip
   ```

3. In Colab choose **Runtime → Change runtime type → GPU**, then run the notebook.
   Leave `REPO_URL` empty and upload `SEANA_BBBC048_Colab.zip` when prompted.

The bundle contains the notebook, experiment helpers, licenses, BBBC048
captures and the 54.3 MB dataset archive. SEANA itself is installed from PyPI,
so the bundle excludes `seana.py` and `_seana_api.py`. It uses an explicit file allowlist and
omits private review snapshots, credentials, generated runs, model weights and
the fetal dataset. It does not require Git LFS. Users of an already published
repository can instead enter its public HTTPS GitHub URL in the first cell or
upload GitHub's **Download ZIP**. The first cell verifies bundle checksums; dataset
extraction independently checks the pinned dataset hash and all image members.

The README's Colab badge currently opens Colab's home page because no public
repository URL is assigned. Once the repository exists, point the badge directly
to `https://colab.research.google.com/github/OWNER/REPO/blob/BRANCH/SEANA_BBBC048_Demo.ipynb`
and set the notebook's default `REPO_URL` to that repository. The upload path
continues to work without that change.

## What the demo measures

The quick preset uses 8,192 training cells sampled proportionally within the
fixed seed-0 training split, and all 6,453 test cells. It uses 500 VST steps,
50 SSL epochs and 20 linear-probe epochs per arm. The long preset uses all 25,813 training cells
and 5,000 VST steps, with the same 50/20 epoch defaults. The notebook's `SSL_EPOCHS`
and `PROBE_EPOCHS` controls let users change these budgets independently of the
data preset. Both use ResNet-18 and
cached/shared SEANA estimates with independent fresh noise after cropping. This
view order is an explicit demo choice, distinct from the main `paper` order.

The two arms receive the same acquired cells, split, random backbone initialization,
SSL method and optimization budget. Estimator fitting is restricted to training
cells. Both probes use acquired images with frozen backbones/BatchNorm. Features
are extracted once and cached, then a final-epoch linear head is evaluated without
selecting an epoch on test scores. Model-initialization hashes are checked for equality.
Changing the preview slider does not mutate the fit or consume training RNG state.

BBBC048 has no paired clean targets; the notebook does not report PSNR or claim
that smoother estimates necessarily preserve more biology. The measured downstream
gain can be small, zero or negative in a short run. Results have
`status=demo_not_benchmark`, remain outside benchmark summaries and retain all
signed deltas. Rare phases have very small test support, shown explicitly.
Historical results are a separate plot with source names and provenance caveats.

## Dependencies, runtime and outputs

Use a GPU for fitting/training. [Colab resource availability and duration vary](https://research.google.com/colaboratory/faq.html);
no fixed completion time is promised. The installer retains Colab's matching
PyTorch/torchvision and NumPy versions and installs the remaining
[demo requirements](../requirements-colab.txt) under constraints. It does not
force the release's PyTorch 2.5.1 pins over the hosted runtime. Versions are saved
with results; [Colab's runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html)
can differ from the locally checked Python 3.12 environment.

The setup installs `seana-ssl==0.1.3` from PyPI and checks that both SEANA modules
load from that installation, even when the notebook is opened inside a checkout.
If a checkout's SEANA module was already imported, restart the runtime and rerun
setup. The VST fit uses the package's `SEANA.fit()` API and saves a library checkpoint
that can be restored with `SEANA.load("path/to/treatment.pt", device="cuda")`.
The preview and cached training estimates use `SEANA.denoise()`. The demo's
existing shared view builder resamples those estimates through the fitted VST.
Package versions and the actual imported source hashes are recorded with results.

The first fit downloads pretrained DRUNet through DeepInverse. It is frozen
throughout; only the VST and then SSL/probe parameters are learned. The images
retain CC BY-NC-SA 3.0, with the archive's attribution/license included in the
bundle. Model/code terms remain in the root third-party notices.

Generated files live under `runs/colab/` with a new directory per experiment.
The small results ZIP excludes weights and cached image arrays; those remain in
the experiment directory. Colab storage is temporary, so download results you
want to retain. Reexecuting a completed comparison in the same directory is
rejected; rerun the experiment-controls cell to start a fresh one.
