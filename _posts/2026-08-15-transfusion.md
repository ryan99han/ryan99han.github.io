---
layout: post
title: Transfusion
tags: [Papers]
---

[Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/pdf/2408.11039)

Transfusion trains a single transformer on both text and images using a different objective for each: next token prediction for text and diffusion for images.

---

## Data
- Each text string is tokenized into a sequence of discrete tokens from a fixed vocabulary
- Each image is encoded via a variational autoencoder (VAE) as latent patches
  - A [1024, 1024, 3] image is encoded into a [128, 128, 8] latent map
  - Each latent patch of [1, 1, 8] represents [8, 8, 3] in pixel space
- Each image sequence is surrounded by beginning of image (BOI) and end of image (EOI) tokens before inserting into the text sequence

## Model Architecture
- A single transformer takes a sequence of high-dimensional vectors as input and produces similar vectors as output
- For text, embedding matrices convert each input integer to vector space
- For images, use either (1) a simple linear layer or (2) up and down blocks of a U-Net

## Transfusion Attention
- Causal attention for every element in the sequence
- Bidirectional attention within the elements of each individual image
- Image patches can attend to every other patch within the same image, but only attend to text or patches of other images which appeared previously

## Training Objective
- Add noise $\epsilon$ to each input latent image $x_0$ according to the diffusion process to produce $x_t$ before patchification

$$L_{Transfusion} = L_{LM} + \lambda \cdot L_{DDPM}$$

## Inference
- In LM mode, sample token by token from the predicted distribution
- When a BOI token is sampled, switch to diffusion mode and append pure noise $x_T$ in the form of $n$ image patches to the input sequence, and denoise over $T$ steps
- At each step $t$, take the noise prediction and use it to produce $x_{t-1}$, which replaces $x_t$ in the sequence
- Once the diffusion process ends, append an EOI token and switch back to LM mode
