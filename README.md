# Ring Galaxy Detection in Euclid Images Using Convolutional Neural Network and ResNet50

Ring galaxies are rare galaxies that have a bright center, like many other galaxies, but they have a bright ring of stars circling them far from the center. They likely form when one galaxy collides directly with another, sending a wave outward that creates a ring of new stars. Studying them helps us understand how galaxies interact and change over time. In the past, galaxy shapes were classified manually by people looking at images. The Galaxy Zoo project used many volunteers to label galaxies. These labels are values between 0 and 1, derived from how many people voted yes or no to the image. The Euclid space telescope is a recent mission that will image billions of galaxies. Their images are high quality, but the image dataset is huge to check manually. In this project, a convolutional neural network (CNN) is trained on Galaxy Zoo data to predict ring galaxy probability. The model is then applied to Euclid images, where it predicts high-probability ring galaxies.

---

## What I did

I took a pretrained ResNet50 and fine-tuned it on ~20,000 Galaxy Zoo images, each labeled with a ring probability voted on by volunteers. The model learns to output a single score per image.                                         

The trickiest bug early on: the model kept predicting the same value for every image. Turned out I was normalizing images to [0,1], but ResNet50 expects a specific ImageNet normalization — using the wrong one completely broke learning. I also had to unfreeze the last 30 layers so the model could actually adapt to galaxy images instead of staying stuck on ImageNet features.

Once trained, I ran it on 2,378 Euclid telescope images (no labels) to rank them by predicted ring probability.

---

## Results

- Test MAE of **0.156** on the validation set of Galaxy Zoo images
- **1,044 out of 2,378** Euclid galaxies scored above 0.5
- Most top-ranked images visually show ring-like structures with a few false positives (edge-on galaxies, spirals), which is expected since the model was trained on SDSS images, and Euclid images look noticeably different

Mild overfitting appeared after epoch 6, where validation loss plateaued while training loss kept dropping. Early stopping would fix this in a future run.

---

## Files

```
python_script/ring_galaxy_project.ipynb   — full code
report/310_final_report.pdf               — full report
images/                                   — result plots
```

## Stack

Python, TensorFlow/Keras, HuggingFace Datasets, NumPy, Matplotlib
