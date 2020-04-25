# GAN
> There are the variety of  Keras implementations of Generative Adversarial Networks (GANs). These examples are simplified versions whose applied on the MNIST which is a handwritten digital dataset. But these ".py" files can be also exploited asthe efficient model-building API. Every one who is interested about GANs can refer and use these implementations to build various kinds of GAN effiently and easily.

## Table of Contents
>* Requirements of Environment
>* Implementations
>	+ Deep Convolutional GAN (DCGAN)
>	+ Conditional GAN (CGAN)
>	+ Wasserstein GAN (WGAN)
>	+ Least-squares GAN (LSGAN)
>	+ Auxiliary Classifier GAN
>	+ InfoGAN
>	+ StackedGAN
>	+ CycleGAN
>	+ Bidirectional GAN
>	+ Boundary-Seeking GAN
>	+ Contex-Conditional GAN
>	+ Coupled GAN
>	+ DiscoGAN
>	+ DualGAN
>	+ Pix2Pix
>	+ PixelDA
>	+ Semi-Supervised GAN
>	+ Super-Resolution GAN

## Requirements of  Environment
> 0. Ubuntu version: 18.04
> 1. Python version: 3.6.9
> 2. Tensorflow version: 1.14.0
> 3. Keras version: 2.3.1
> 4. CUDA version: 10.0
> 5. Cudnn version: 7.4.1.5
> 6. TensorRT version: 6.0.1.5
> 7. GPU driver and GPU model: 430.50 & GeForce GTX 1060 6G

## Implementations
### Deep Convolutional GAN (DCGAN)
> DCGAN could be implemented using deep CNNs.


### Conditional GAN (CGAN)
>CGAN added the condition (indexing label) network to DCGAN, and CGAN can generate the specific label fake data as your appointment.

### Wasserstein GAN (WGAN)
>WGAN apply the wasserstein loss to promote the stability of training.

### Least-square GAN (LSGAN)
>There are two problems being addressend on Wasserstein GAN paper, they are concerned the stability of training and the generative quality. But Wasserstein GAN only solve the convergent stability problem. LSGAN apply the least-square losss to solve these two problems about GAN. 

### Auxiliary Classifier GAN (ACGAN)
> ACGAN is similar in principle to CGAN. For both CGAN and ACGAN, the generator inputs are noise and its label. The output is a fake image belonging to the input class label. For the generator, CGAN only detect input wheather be real or fake. And ACGAN not only do it, but also classify its class.