# SparseGS-W

Sparse-view novel view synthesis based on 3D Gaussian Splatting. The training pipeline uses
diffusion-guided pseudo-view supervision and jointly optimizes camera poses.

## Pipeline

1. **Fine-tune Stable Diffusion on the sparse images (DreamBooth).**
   Fine-tune a Stable Diffusion UNet on the sparse training images with
   [DreamBooth](https://dreambooth.github.io/). Pass the DreamBooth output directory
   (containing `unet/`) to the training script through
   `--use_finetune_unet <dreambooth-unet-path>`.

2. **Reconstruct the scene with DUSt3R.**
   Run [DUSt3R](https://github.com/naver/dust3r?tab=readme-ov-file) on the same sparse images
   to recover the camera intrinsics/extrinsics and an initial point cloud. Save them in the
   dataset layout described in [Data](#data) (`train_intrinsic.txt`, `train_c2w.pkl` or
   `train_c2w.txt`, `test_intrinsic.txt`, `test_c2w.txt`, and
   `dust3r_pc_with_sky.ply`).

3. **Train 3DGS.**
   ```shell
   python train.py -s <dataset> -m <output> --eval --dataset_name NerfW --n_sparse <num_views> \
       --use_finetune_unet <dreambooth-unet-path>
   ```

4. **Render.**
   ```shell
   python render.py -m <output>
   ```

5. **Evaluate.**
   ```shell
   python metrics.py -m <output>
   ```

## Requirements

- Python 3.9+ and CUDA 11.8. The base environment follows the original
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) repository.

## Data format

```
<dataset>
|-- images
|-- train_intrinsic.txt
|-- train_c2w.txt
|-- test_intrinsic.txt
|-- test_c2w.txt
|-- test/
|-- dust3r_pc_with_sky.ply
```

`*_intrinsic.txt` stores camera intrinsics, and `*_c2w.txt` stores camera-to-world
extrinsics.

`--n_sparse` must equal the number of training views.

## Training

```shell
python train.py -s <dataset> -m <output> --eval --dataset_name NerfW --n_sparse 4 \
    --use_finetune_unet <dreambooth-unet-path>
```

The Stable Diffusion checkpoint (`runwayml/stable-diffusion-v1-5`) is downloaded from
Hugging Face on first use.

Useful options:

```shell
python train.py -s <dataset> -m <output> --eval --dataset_name NerfW --n_sparse 4 \
    --use_finetune_unet <dreambooth-unet-path> \
    --iterations 6500 --use_CL_training_strategy
```

## Rendering

```shell
python render.py -m <output>
```

`render.py` writes train, novel, pseudo and test renderings. Test views additionally undergo
pose optimization; its result is stored under `test/ours_<iteration>/opt_pose_render<steps>`.

## Evaluation

```shell
python metrics.py -m <output>
```
