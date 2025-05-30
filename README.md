# Open-AI-CLIP-implementation-Open AI CLIP implementation  using Flicker 8k dataset

# CLIP: Contrastive Language-Image Pretraining in PyTorch

This repository contains an end-to-end implementation of a simplified CLIP (Contrastive Language-Image Pretraining) model using PyTorch. The model learns joint embeddings of images and their corresponding text captions, enabling cross-modal retrieval tasks such as retrieving the most relevant images given a text query.

## Overview

This implementation uses ResNet50 as the image encoder and DistilBERT as the text encoder. The encoded image and text features are projected into a shared embedding space where contrastive learning is used to align semantically similar image-text pairs. The model is trained using a symmetric cross-entropy loss computed from dot-product similarities.

## Features

- ResNet50-based image encoder via timm library
- DistilBERT-based text encoder from HuggingFace Transformers
- Tokenization of captions with automatic padding and truncation
- Data augmentation using Albumentations
- Custom contrastive loss function
- Training and validation routines
- Inference and image retrieval using text queries

## Installation

Install the required dependencies using pip:


## Configuration

Model and training parameters can be set using the `CFG` class. Important parameters include image size, learning rates, number of epochs, projection dimension, and batch size. The image and caption dataset paths are also configured here.

## Dataset Format

The dataset should include a CSV file (e.g., `captions.csv`) with columns `id`, `image`, and `caption`. The image files referenced must be present in the directory specified by `CFG.image_path`. Multiple captions per image are supported by repeating the image name in the CSV.

## Model Components

### Image Encoder

The image encoder is based on ResNet50. The encoder outputs a 2048-dimensional feature vector for each image, which is then passed through a projection head to map it into a 256-dimensional embedding space.

### Text Encoder

The text encoder uses DistilBERT. The CLS token's hidden state (a 768-dimensional vector) is extracted as the sentence representation and passed through a projection head to obtain a 256-dimensional embedding.

### Projection Head

Both image and text encoders are followed by a projection head. This head is a two-layer neural network with GELU activation, dropout, and layer normalization, projecting features into a 256-dimensional shared space.

### Contrastive Loss

The contrastive loss is computed by taking the dot product between image and text embeddings and applying a softmax. A symmetric loss is used: the loss is calculated for both image-to-text and text-to-image directions and averaged.

## Training

The model is trained using AdamW optimizer and ReduceLROnPlateau scheduler. The training loop includes standard forward, backward, and optimization steps. The model is evaluated on a validation set each epoch and the best model is saved based on validation loss.

## Inference

The inference pipeline uses the trained model to generate embeddings for validation images. Given a text query, it generates its embedding, computes cosine similarity with the image embeddings, and retrieves the top-k most similar images.

## Example Usage

To train the model:

```python
main()
```

To perform inference:

```python
find_matches(model,
             image_embeddings,
             query="a group of people dancing in a party",
             image_filenames=valid_df['image'].values,
             n=9)
```

## Notes

- The training code supports GPU acceleration.
- The model was trained on the Flickr8k dataset.
- You can add additional data augmentations or use different encoders by modifying the configuration.

## References

- OpenAI CLIP Paper: https://openai.com/research/clip
- HuggingFace Transformers: https://huggingface.co/transformers/
- Timm Image Models: https://github.com/rwightman/pytorch-image-models


