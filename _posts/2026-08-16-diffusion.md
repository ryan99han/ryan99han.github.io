---
layout: post
title: Diffusion Models
tags: [Papers]
---

[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

Diffusion models learn to reverse a gradual noising process, converting pure noise into samples from a complex data distribution by learning to predict and remove noise at each step.

---

## Forward Process
![Diffusion](/assets/images/diffusion.png)
- The intuition is that we add noise to images so that we can convert the unknown and complex distribution that our training set belongs to into a simple, chosen distribution that is easy to sample from
- Let $x_0$ be the original image and $x_T$ follow an isotropic Gaussian (a Gaussian distribution in which variance is equal in all directions)
- We define a forward noising process $q$ which produces latents $x_1$ to $x_T$

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$$

- $x_t$ is the result of the forward process at step $t$, $\sqrt{1-\beta_t}\,x_{t-1}$ is the mean, and $\beta_t I$ is the covariance matrix
- $\beta_t$ is the variance schedule, and ranges from 0 to 1. The values are kept low to prevent variance from exploding
- Using the reparameterization trick

$$\mathcal{N}(\mu, \sigma^2) = \mu + \sigma \cdot \epsilon$$

$$x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon$$

- At each step, we do two things:
  - Scale down the signal: take the previous image $x_{t-1}$ and scale it down by $\sqrt{1 - \beta_t}$ to fade the original signal
  - Add noise: add a random amount of noise $\sqrt{\beta_t}\,\epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$

### Simplification
- For training, we use samples from an arbitrary timestep $t$, which means we would need to iterate through $t$ steps to generate one training sample
- We define the entire noise to be added at $T$ as

$$q(x_1, \dots, x_T \mid x_0) \coloneqq \prod_{t=1}^T q(x_t \mid x_{t-1})$$

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$$

- To begin simplifying this, let

$$\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t \coloneqq \prod_{s=1}^t \alpha_s$$

- Then we have

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}\,x_{t-1}, (1 - \alpha_t) I)$$

- We can use the reparameterization trick

$$x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1 - \alpha_t}\,\epsilon_{t-1}$$

- For $t = 1$

$$x_1 = \sqrt{\alpha_1}\,x_0 + \sqrt{1 - \alpha_1}\,\epsilon_0$$

- For $t = 2$

$$x_2 = \sqrt{\alpha_2}\,x_1 + \sqrt{1 - \alpha_2}\,\epsilon_1$$

- Substituting $x_1$

$$x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1}\,x_0 + \sqrt{1 - \alpha_1}\,\epsilon_0) + \sqrt{1 - \alpha_2}\,\epsilon_1$$

$$x_2 = \sqrt{\alpha_2 \alpha_1}\,x_0 + \sqrt{\alpha_2 (1 - \alpha_1)}\,\epsilon_0 + \sqrt{1 - \alpha_2}\,\epsilon_1$$

$$x_2 = \sqrt{\bar{\alpha}_2}\,x_0 + \sqrt{\alpha_2 (1 - \alpha_1)}\,\epsilon_0 + \sqrt{1 - \alpha_2}\,\epsilon_1$$

- The terms involving the noise, $\epsilon_0$ and $\epsilon_1$, are from independent Gaussian distributions
- The sum of two independent Gaussian random variables is also a random Gaussian variable, with a mean equal to the sum of the means and a variance equal to the sum of the variances
- Since each $\epsilon$ is sampled from a normal Gaussian distribution with mean zero, the sum of the means is also zero. Therefore, we are left with the following mean

$$\mu = \sqrt{\bar{\alpha}_2}\,x_0$$

- Now we calculate the combined variance for the noise terms, noting the following properties of variance for independent variables

$$Var(cX) = c^2 Var(X), \qquad Var(X + Y) = Var(X) + Var(Y)$$

- Recall that $Var(\epsilon_i) = I$ because the noise we add is from a normal Gaussian distribution with variance 1

$$Var(x_2) = Var(\sqrt{\bar{\alpha}_2}\,x_0) + Var(\sqrt{\alpha_2 (1 - \alpha_1)}\,\epsilon_0) + Var(\sqrt{1 - \alpha_2}\,\epsilon_1)$$

$$= Var(\sqrt{\alpha_2 (1 - \alpha_1)}\,\epsilon_0) + Var(\sqrt{1 - \alpha_2}\,\epsilon_1)$$

$$= \alpha_2 (1 - \alpha_1) \, Var(\epsilon_0) + (1 - \alpha_2) \, Var(\epsilon_1)$$

$$= \alpha_2 (1 - \alpha_1) + (1 - \alpha_2)$$

$$= \alpha_2 - \alpha_2 \alpha_1 + 1 - \alpha_2$$

$$= 1 - \alpha_2 \alpha_1$$

$$\sigma^2 = 1 - \bar{\alpha}_2$$

- For any time between $t = 0$ to $t = T$, we have

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\,x_0, (1 - \bar{\alpha}_t) I)$$

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$$

$$\bar{\alpha}_t \coloneqq \prod_{s=1}^t (1 - \beta_s)$$

### Application
- What does this mean at the pixel level?

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\,x_0, (1 - \bar{\alpha}_t) I)$$

- Remember that in a multivariate normal distribution, the covariance matrix $\Sigma$ describes the variance of each individual dimension and the covariance between dimensions
  - Diagonal elements of $\Sigma$ give the variance for each pixel channel
  - Off-diagonal elements of $\Sigma$ give the covariance between pairs of pixel channels
- The covariance is zero, meaning that the amount of noise we sample and add to one pixel is completely independent of the noise sampled and added to another pixel. Therefore, random noise is applied independently across the whole image
- We take a single image $x_0$ from our training data with shape (1024, 1024, 3)
- We then choose a random timestep $t \in [1, T]$ and directly calculate $\bar{\alpha}_t$ based on the noise schedule

$$\bar{\alpha}_t \coloneqq \prod_{s=1}^t (1 - \beta_s)$$

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$$

- Let the red channel value of a single pixel be $x_{0,red}(i, j)$, which is a floating point value scaled between -1 and 1
- We sample a value $\epsilon$ from the standard normal distribution $\mathcal{N}(0, 1)$ for each pixel channel independently, and then compute the following

$$x_{t, red}(i, j) = \sqrt{\bar{\alpha}_t}\,x_{0, red}(i, j) + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$$

## Reverse Process
- The goal of the reverse diffusion process, denoted $p_\theta$, is to learn the reverse of the forward noising process
- Starting from a sample of pure noise $x_T \sim \mathcal{N}(0, I)$, the model iteratively denoises $x_T$ to generate a clean data point $x_0$

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)$$

- The core of the reverse process is learning the transition conditional probability $p_\theta(x_{t-1} \mid x_t)$
- If the noise added in the forward process is sufficiently small at each step, the reverse transition can also be approximated by a Gaussian distribution

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

- The challenge lies in estimating the mean $\mu_\theta(x_t, t)$ and variance $\Sigma_\theta(x_t, t)$
- A critical insight in training diffusion models is that the true reverse distribution $p(x_{t-1} \mid x_t)$ is computationally intractable, since it requires knowing the distribution of all possible clean images
- However, the posterior distribution of the forward process $q(x_{t-1} \mid x_t, x_0)$ is tractable and can be expressed in closed form
  - The posterior distribution of the forward process represents the probability distribution of the slightly less noisy data at the previous timestep $t-1$ given the noisier data at the current timestep $t$ and the original clean data $x_0$
  - A posterior distribution provides a way to combine a prior distribution with observed data to arrive at an updated conclusion
- This posterior provides a target distribution for the neural network to learn, and the model is trained to minimize the difference between the learned distribution $p_\theta(x_{t-1} \mid x_t)$ and the true conditional $q(x_{t-1} \mid x_t, x_0)$
- In practice, the network is trained to predict the noise at each step, and this predicted noise is used to estimate the mean and variance of $q(x_{t-1} \mid x_t, x_0)$, effectively learning how to perform a single, correct denoising step
- By training on all possible timesteps $t$ and all data samples $x_0$, the network generalizes this ability and learns to denoise random input from scratch without needing the original image as a guide

$$q(x_{t-1} \mid x_t) = q(x_t \mid x_{t-1}) \frac{q(x_{t-1})}{q(x_t)}$$

$$q(x_{t-1} \mid x_t, x_0) = q(x_t \mid x_{t-1}, x_0) \frac{q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

- After a good deal of math, we arrive at

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}x_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t$$

- The primary use of $q(x_{t-1} \mid x_t, x_0)$ is to define the Variational Lower Bound (VLB) loss function
- Diffusion models are latent variable models, and their training objective is to maximize the log-likelihood of the data $\log(p_\theta(x_0))$ by optimizing the Evidence Lower Bound (ELBO), $\mathcal{L}$
- The ELBO can be decomposed into a sum of Kullback-Leibler (KL) divergences for each timestep $t$

$$L \geq \sum_{t=1}^T \mathbb{E}_{x_0, x_t}\Big[D_{KL}\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big)\Big] + \dots$$

- Here, $q(x_{t-1} \mid x_t, x_0)$ is the closed-form posterior, which acts as the target distribution for the KL divergence. It's the "true" way to reverse one step of diffusion, conditioned on the original, clean data $x_0$
- $p_\theta(x_{t-1} \mid x_t)$ is the learned reverse transition produced by the neural network (parameterized by $\theta$)
- The objective is to train the parameters $\theta$ of the neural network to minimize the KL divergence, effectively forcing the learned reverse distribution $p_\theta$ to be as close as possible to the ideal target $q$

$$\min_\theta \sum_{t=1}^T \text{Distance}\big(q(x_{t-1} \mid x_t, x_0), p_\theta(x_{t-1} \mid x_t)\big)$$

- Because both $q(x_{t-1} \mid x_t, x_0)$ and $p_\theta(x_{t-1} \mid x_t)$ are designed to be Gaussian distributions, they are defined entirely by their mean $\mu$ and covariance $\Sigma$. From above, we know the exact expressions for $\tilde{\mu}_t$ and variance $\tilde{\Sigma}_t$
- In the popular DDPM formulation, the mean $\tilde{\mu}_t$ of the target posterior $q(x_{t-1} \mid x_t, x_0)$ can be expressed as a function of the noise added to create $x_t$ from $x_0$. The network is then re-parameterized to predict the noise $\epsilon$ instead of the mean $\mu$ directly
- Specifically, the neural network $\epsilon_\theta(x_t, t)$ is trained to predict the noise $\epsilon$ that was used to generate $x_t$ from $x_0$
- This leads to a simpler and more stable mean squared error loss based on noise prediction

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\Big[\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2\Big]$$

### Training
- For each training step, sample a clean image $x_0$
- Randomly choose a timestep $t$ between 1 and $T$
- Apply the forward noising process to generate $x_t$ from $x_0$ by adding noise $\epsilon$
- The neural network takes the noisy image $x_t$ and timestep $t$ as input and predicts the noise $\epsilon_\theta(x_t, t)$ that was added at time $t$
- The loss function (e.g. MSE) is calculated between the predicted noise $\epsilon_\theta$ and the actual noise $\epsilon$
- By minimizing this loss over millions of examples, the network learns to accurately predict the noise component for any noisy image at any timestep, which approximates the complex reverse diffusion transition needed to go from $x_t$ back to a slightly less noisy $x_{t-1}$

### Inference
- Start with $x_T$, a tensor of pure, random Gaussian noise
- From $t = T$ to $t = 1$, the neural network does the following:
  - Takes the current noisy sample $x_t$ and current timestep $t$
  - Predicts the noise $\epsilon_\theta$ in $x_t$
  - Calculates and subtracts a weighted portion of this predicted noise from $x_t$ to obtain a slightly less noisy sample $x_{t-1}$
- After $T$ steps of iterative denoising, the final output $x_0$ is a clean, synthesized image that closely matches the distribution of the original training data
