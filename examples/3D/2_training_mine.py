from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D, StarDistData3D

################### DATA

X = sorted(glob('data/train/images/*.tif'))
Y = sorted(glob('data/train/masks/*.tif'))
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]

################### Normalize images and fill small label holes
axis_norm = (0, 1, 2)  # normalize channels independently
# axis_norm = (0,1,2,3) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

################### Split into train and validation datasets.
# assert len(X) > 1, "not enough training data"
X_val, Y_val = [X[1]], [Y[1]]
X_trn, Y_trn = [X[0]], [Y[0]]
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

################### Display
i = 0
img, lbl = X[i], Y[i]
assert img.ndim in (3, 4)
img = img if img.ndim == 3 else img[..., :3]
z = img.shape[0] // 2
lbl_cmap = random_label_cmap()
plt.figure(figsize=(16, 10))
plt.subplot(121);
plt.imshow(img[z], cmap='gray');
plt.axis('off');
plt.title('Raw image (XY slice)')
plt.subplot(122);
plt.imshow(lbl[z], cmap=lbl_cmap);
plt.axis('off');
plt.title('GT labels (XY slice)')
None;

################### CONFIG
# print(Config3D.__doc__)
####### A StarDist3D model is specified via a Config3D object.
extents = calculate_extents(Y)
anisotropy = tuple(np.max(extents) / extents)
print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

# 96 is a good default choice (see 1_data.ipynb)
n_rays = 64  # 96

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

conf = Config3D(
    rays=rays,
    grid=grid,
    anisotropy=anisotropy,
    use_gpu=use_gpu,
    n_channel_in=n_channel,
    # adjust for your data below (make patch size as large as possible)
    train_patch_size=(48, 96, 96),
    train_batch_size=2,
)
print(conf)
vars(conf)

###################
if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory

    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)

model = StarDist3D(conf, name='stardist', basedir='models')

# Check if the neural network has a large enough field of view to see up to the boundary of most objects.
median_size = calculate_extents(Y, np.median)
fov = np.array(model._axes_tile_overlap('ZYX'))
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

################### Training
augmenter = None

quick_demo = False;  # True
if quick_demo:
    print(
        "NOTE: This is only for a quick demonstration!\n"
        "      Please set the variable 'quick_demo = False' for proper (long) training.",
        file=sys.stderr, flush=True
    )
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter,
                epochs=2, steps_per_epoch=5)

    print("====> Stopping training and loading previously trained demo model from disk.", file=sys.stderr, flush=True)
    model = StarDist3D(None, name='3D_demo', basedir='../../models/examples')
    model.basedir = None  # to prevent files of the demo model to be overwritten (not needed for your model)
else:
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    # model.fit(X_trn, Y_trn, epochs=1, batch_size=1) # error: StarDist3D' object has no attribute 'fit'
None;
