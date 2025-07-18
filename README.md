# Galaxy Merger Classification with CNNs

This is the final project for the Signal and Imaging Acquisition and Modelling in Environment course at Unimib.

## Summary

### Objective

The goal of this project is to automate the classification of **galaxy mergers** using deep learning, specifically convolutional neural networks (CNNs).
Galaxy mergers are crucial to understanding cosmic evolution, affecting star formation, black hole activity, and overall galactic structure.

---

### Approach

The project uses simulated galaxy image data from the **Illustris-1** cosmological simulation. Each sample contains three image channels corresponding to HST filters, and is labeled based on whether the galaxy is expected to merge within the next 500 Myr.

The approach includes:

* **Dataset preparation**:

  * Using both pristine and noise-augmented versions of the data to simulate real observational conditions.
  * Addressing concerns of potential data leakage due to how the original dataset was constructed.

* **Data augmentation**:

  * Applied simple, non-destructive augmentations (random flips and 90° rotations) to the training set only.

* **Denoising** (core focus of this project):
  A substantial portion of the work focused on investigating denoising techniques to handle realistic astronomical noise, which can obscure morphological features relevant to classification. The goal was to pre-process images in a way that reduces noise without destroying critical information. The denoising methods explored include:

  * **Fourier-based filtering**: Removes high-frequency components associated with noise while preserving low-frequency structure.
  * **Morphological operations**:

    * *White top-hat transform* to enhance compact bright features.
    * *Minimum filter* to estimate and subtract background light.
  * **Autoencoder denoising**: Implemented a shallow U-Net architecture trained to reconstruct clean images from noisy inputs, leveraging self-supervised learning principles.

  These methods were evaluated both qualitatively and by measuring entropy reduction, under the assumption that denoised images should exhibit less randomness while retaining galaxy structure.

* **Model design**:
  Three CNNs were implemented and evaluated:

  * A custom architecture (**FastHeavyCNN**) designed for efficient training and inference (don't judge my poor naming skills, i'm sadly aware of it)
  * A pre-trained **ResNet18** fine-tuned on the dataset
  * The baseline **DeepMerge** architecture from prior literature

All models were trained with **AdamW** optimizer and **Cosine Annealing** learning rate schedule. A **weighted binary cross-entropy** loss was used to handle mild class imbalance.

---

See the full report for a detailed breakdown of architecture, denoising methods, and further considerations.
