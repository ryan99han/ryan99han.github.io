---
layout: post
title: Diffusion Transformers (DiT)
tags: [Papers]
---

[Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748)

Diffusion transformers are a new class of models which replace the common U-Net backbone with a transformer that operates on latent patches

---

## DiT

![DiT](/assets/images/dit.png)

### Overview
- The first layer of the DiT patchifies the input image into $T$ tokens
- Standard ViT frequency-based positional embeddings (sine-cosine) are applied to all input tokens
- The best performing DiT block swaps the standard layer norm layers in the transformer blocks with adaptive layer normalization layers
- Instead of directly learning dimension-wise scale and shift parameters $\gamma$ and $\beta$, they are regressed from the sum of the embedding vectors of $t$ and $c$
- Since prior work on ResNets has found that initializing each residual block as the identity function is beneficial, in addition to regressing $\gamma$ and $\beta$, the dimension-wise scaling parameters $\alpha$ that are applied immediately to any residual connections in a DiT block are also regressed
- The MLP is initialized to output the zero-vector for all $\alpha$, initializing the full DiT block as the identity function
- A sequence of $N$ DiT blocks are used, ranging from 12 to 28
- After the final DiT block, the sequence of image tokens needs to be decoded into an output noise prediction and an output diagonal covariance prediction
- A final adaptive layer norm is applied before a linear decoder decodes each token
- The decoded tokens are rearranged into their original spatial layout to get the predicted noise and covariance
