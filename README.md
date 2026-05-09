# Ring Galaxy Detection in Euclid Images

PHYS 310 Final Project — Nam Bui

Ring galaxies form when two galaxies collide head-on, sending a shockwave outward that creates a bright ring of new stars. They're rare, visually striking, and hard to find manually at scale. This project trains a CNN to score galaxy images by ring probability, then applies it to images from the Euclid space telescope to surface the most likely candidates.

The full write-up is in `report/310_final_report.pdf`.

---

## What I did

I fine-tuned a pretrained ResNet50 on ~20,000 labeled images from the Galaxy Zoo `gz_rings` dataset, where labels are crowd-sourced ring probability scores between 0 and 1. The model outputs a single probability per image.

One key thing I learned early: using simple [0,1] normalization caused the model to predict a constant value. ResNet50 was pretrained with ImageNet-specific channel normalization, so using the wrong preprocessing completely broke learning. Fixing that (and unfreezing the last 30 layers of ResNet50 so batch norm could adapt) made the model actually train.

The trained model was then run on 2,378 Euclid telescope images, which have no labels, to rank them by predicted ring probability.

---

## Results

- Test MAE of **0.156** on held-out Galaxy Zoo images
- **1,044 out of 2,378** Euclid galaxies scored above 0.5
- Most top-ranked images visually show ring-like structures — a few false positives appear (edge-on galaxies, spirals), which is expected since the model was trained on SDSS images and Euclid images look noticeably different

Mild overfitting appeared after epoch 6 — validation loss plateaued while training loss kept dropping. Early stopping would fix this in a future run.

---

## Files

```
python_script/ring_galaxy_project.ipynb   — full code
trained_model/ring_galaxy_model.keras     — saved model
report/310_final_report.pdf               — full report
images/                                   — result plots
```

## Stack

Python, TensorFlow/Keras, HuggingFace Datasets, NumPy, Matplotlib
