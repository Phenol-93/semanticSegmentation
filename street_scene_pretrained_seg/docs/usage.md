# Usage

This project only runs inference with a local MMSegmentation pretrained checkpoint. It does not train a model and does not require manually annotated masks.

## 1. Prepare Images

Put your original street-scene images into:

```text
input_images/
```

Example:

```text
input_images/
  000001.jpg
  000002.jpg
  000003.png
```

## 2. Prepare Local Model Files

The inference workflow will use a local MMSegmentation config file and a local `.pth` checkpoint file under:

```text
checkpoints/
```

These files will be downloaded explicitly in a later step. Inference scripts should read these local files instead of downloading models during inference.

## 3. Run Inference Later

After the model download and inference scripts are created, the pipeline will read images from `input_images/` and write results to `outputs/`.

No training data, dataset config, or ground-truth mask is needed for this stage.
