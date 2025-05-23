# Image Generator 32x32

32px × 32px image generator trained using simple neural networks on CIFAR-10-style data.

## Features

- Conditional image generation (by class label)
- Simple fully-connected neural network (PyTorch Lightning)
- Training and image generation scripts
- Dataset download script (`download.py`)

## Requirements

- Python 3.8+
- torch
- torchvision
- pytorch-lightning
- matplotlib
- pillow

Install dependencies:
```bash
pip install torch torchvision pytorch-lightning matplotlib pillow
```

## Dataset

You can automatically download and extract the CIFAR-10 dataset using the provided `download.py` script:

```bash
python download.py
```

This will download CIFAR-10 and organize images into the `cifar10_images` folder, ready for training.

**Manual structure (if needed):**
```
cifar10_images/
  airplane/
    1.png
    ...
  automobile/
    ...
  ...
```

## Usage

### Train the Model

```bash
python main.py
```
When prompted, enter `y` to train.

### Generate Images

After training, run:
```bash
python main.py
```
When prompted, enter `n` to skip training.  
Then enter class labels (0-9) separated by spaces to generate images for those classes.

The generated images will be saved as `generated.png`.

**Class labels:**
| Label | Class      |
|-------|------------|
| 0     | airplane   |
| 1     | automobile |
| 2     | bird       |
| 3     | cat        |
| 4     | deer       |
| 5     | dog        |
| 6     | frog       |
| 7     | horse      |
| 8     | ship       |
| 9     | truck      |

## Notes

- The model outputs images in the range `[-1, 1]` and saves them normalized to `[0, 1]`.
- For best results, train for several epochs.
