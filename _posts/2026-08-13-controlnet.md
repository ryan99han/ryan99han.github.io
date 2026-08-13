---
layout: post
title: ControlNet
tags: [Papers]
---

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543)

ControlNet is a neural network architecture which adds spatial conditioning controls to large, pretrained text-to-image diffusion models.

---

## ControlNet
- Spatial controls include edges, depth, segmentation, human pose, etc.
- The parameters of the pretrained diffusion model are locked, and a trainable copy of the encoding layers is made
    - The trainable copy and the original, locked model are connected with zero convolution layers, with weights initialized to zeros so that they progressively grow during training
    - This ensures that harmful noise is not added to the features at the beginning of training

## Architecture
- Let the term *network block* refer to a set of layers commonly put together to form a single unit of a neural network, e.g. a ResNet block, multi-head attention block, transformer block, etc.
- Suppose $F(\cdot;\theta)$ is such a trained network block with parameters $\theta$ which transforms an input feature map $x$ into another feature map $y$ as

$$y = F(x;\theta)$$

- To add a ControlNet to such a pretrained neural block, we freeze the parameters $\theta$ of the original block and also clone the block to a trainable copy with parameters $\theta_c$
- The trainable copy takes an external conditioning vector $c$ as input, and is connected to the frozen model with zero convolution layers denoted $Z(\cdot ; \cdot)$, which is a 1x1 convolution layer with both weight and bias initialized to zeros

$$y_c = F(x;\theta) + Z\Big( F \big( x + Z(c;\theta_{z1}); \theta_c \big); \theta_{z2} \Big)$$

- Breaking it down
    - $Z(c;\theta_{z1})$ is the output of the first zero convolution
    - $x + Z(c;\theta_{z1})$ is the result of adding $x$ to the output of the first zero convolution
    - $F \big( x + Z(c;\theta_{z1}); \theta_c \big)$ is the result of the trainable copy
    - $Z\Big( F \big( x + Z(c;\theta_{z1}); \theta_c \big); \theta_{z2} \Big)$ is the result of the trainable copy fed through a zero convolution
    - $F(x;\theta) + Z\Big( F \big( x + Z(c;\theta_{z1}); \theta_c \big); \theta_{z2} \Big)$ is the result of adding the output from the frozen network block and the ControlNet
    ![ControlNetBlock](/assets/images/controlnet_block.png){: .centered width="40%"}
- In the first training step, since the weight and bias parameters of the zero convolution layers are initialized to zero

$$y_c = y$$

- This method extends to the whole U-Net architecture
![ControlNetArchitecture](/assets/images/controlnet_architecture.png){: .centered width="60%"}


## Training
- ControlNet has the same image diffusion training method and learning objective (predicting noise)
- During training, 50% of text prompts are randomly replaced with empty strings to increase ControlNet's ability to directly recognize semantics in conditioning images (edges, depth, etc.) as a replacement for prompt
- The model is always able to predict high-quality images during training since zero convolutions do not add noise to the network
- The model does not gradually learn the control conditioning but abruptly succeeds, suddenly converging at a checkpoint