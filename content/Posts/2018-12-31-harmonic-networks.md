---
Title: 'Harmonic networks: implementation of paper results'
date: 2019-03-10
analytics: true
slug: harmonic-network/
tags: deep learning,computer vision,work in progress
---
I implement an interesting result from a recent paper on convolutional neural networks.

# Introduction

In this post I will briefly discuss my implementation of a model introduced in [this paper](https://arxiv.org/abs/1812.03205v1).

In short, the authors suggest using predefined filters in a convolutional network based on Discrete Cosine Transform.

I used PyTorch for neural network implementation, the other packages include `pandas` for data reading,
`numpy, scipy, skimage`, etc.

The dataset I was using is the [skin cancer MNIST](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) from Kaggle.

I really enjoyed working on this project! However, it is still far from being complete and I will try to fix some errors it has in due time.

# Paper summary

In this paper, the authors propose a modification to the common algorithm of convolutional neural networks.

Classically, such networks are comprised of convolutional layers. Each layer, in turn, has a corresponding amount of kernels which are slided over the input during convolution ([see visualization here](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)). As the input is carried through the network, in the final, non-convolutional (fully-connected) layer the network measure the error between the true label/value and the prediction is carried backwards through back propagation algorithm. That way the weights (kernels) are adjusted to accommodate for the discrepancy in prediction and improve feature extraction, which is the primary purpose of convolutional layers.

The paper, however, considers a different approach to convolutional neural networks. In suggested algorithm, the net's layers are predefined by decomposition of the input according to the Discrete Cosine Transform (the DCT). The [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform ) is a method for feature extraction that is based on [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform). It separates frequencies that comprise the input signal from itself and thus produce features (the frequency spectrum).

The authors use convolutional  approach to computing the Discrete Cosine transform, building filters that result in equivalent mathematical formulations upon convolution. Using these filters as kernels in CNN, they construct a Harmonic Network. Such network can be computationally expensive (more on that later), but it is able to show state of the art results on common image datasets such as CIFAR10.

The harmonic block that replaces the usual convolutional layer consists of a linear combination of features from the DCT of the input from the previous layer. These features can optionally be batch normalized.

# Difficulties in implementation

Let me discuss some difficulties I ran into when I was implementing the network. I will try to be as brief as I can.

## Finding the right kernel (filter bank)

Firs of all, representing the DCT as a convolution sounds intuitively simple but turned out to be more difficult in practice. Let's look at the formula transforming the 2D N-by-N input signal $x$ into 2D output signal $\hat X$:

$\hat X_{u,v} = \sum_{ii=0}^{N-1}\left[\sum_{jj=0}^{N-1} x_{ii,jj} \cos\left(\frac{\pi}{N}\left(ii+0.5\right) u \right)\right] \cos\left(\frac{\pi}{N}\left(jj+0.5\right) v \right).$

So the approach is to use a separate kernel for each combination of $(u,~v)$ indices. They will implicitly represent the direction in which our filter is looking at the image.

So, if our sliding convolution window is $N\times N$ then we need $N^2$ filters, $N\times N$ each:

```py
import torch
import numpy as np
PI = np.pi
def fltr(u, v, N, k): # note, that I will always use N=k and N=K in where
                      # but we can have N>K. I have not tried N<K,
                      # it'd be pretty cool to try that as well
  return torch.as_tensor([[torch.cos(torch.as_tensor(v*PI/N*(ii+0.5)))
                           * torch.cos(torch.as_tensor(u*PI/N*(jj+0.5)))
                           for ii in range(k)] for jj in range(k)])
```

So, once we get the necessary filters, we need to properly collect them into the so-called *filter bank*.

```py
import torch
def get_filter_bank(input_channels, N, K):

  filter_bank = torch.stack([torch.stack([fltr(j, i, N, K)
                                          for i in range(K)]) for j in range(K)])
  filter_bank = filter_bank.reshape([-1, 1, K, K])
  filter_bank = torch.cat([filter_bank]*input_channels, dim=1)
  filter_bank = filter_bank.to('cuda')
  return filter_bank
```

Great, the filters are collected. The cool thing with PyTorch is that by default these tensors will not be updated in the backwards pass. This is because the property called `requires_grad` is initialized to `False` automatically. (I should probably add that it can always be manipulated manually, but we will not worry about that here.)

## Further processing of harmonic blocks

One convolution with kernels is not enough. The authors propose a way of combining them linearly through $1\times1$ convolution. That way each consequent layer is a linear combination of the previous one. This convolution __is__ affected by the backwards pass. The linear combination occurs across the result of convolution with DCT filters.

If the input has 3 channels and each channel produces 9 convolution results ($3\times 3$ filters, 9 of them)
then we get the output shape after the harmonic block as (3,9,W,H) where W and H are width and height respectively. The linear combination than happens across the __9__ outputs for each channel. That's it!

## Sending to GPU device using `torch.cuda()`

A problem I encountered while implementing the Harmonic block of the network was that just sending the model to GPU using

```py
model.cuda()
```

is not enough. I had to "manually" send the convolution weights. Note that it's not really manually as in "low-level" but rather this line of code from before:

```py
filter_bank = filter_bank.to('cuda')
```

Not sure why the model does not send the DCT filters upon regular GPU sending but manually it all works.

## Selecting learning rate

Another problem was the learning rate. From lessons on [FastAI](https://fast.ai) I learned that trying the learning rate of 0.01 is quite common (the amazing function for finding a proper learning rate consistently recommends doing so across multiple CNN models). However, in here, we get quite rapid loss increase!

### Update, March,2019

Experimenting with learning rate finding yielded that 0.001 learning rate with *Adam* optimizer works really well for my architecture, especially on an unbalanced [dataset](https://arxiv.org/abs/1803.10417) I use for my research.

## Image preprocessing, added March,2019

In the preprocessing stage, the images are downsized to 64 by 64 pixels using `skimage` library. Originals in the dataset used are 450 by 600, which can take too much of the memory.

Additional preprocessing consists of hair removal described in [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.3821&rep=rep1&type=pdf).

# Result

Although it overfits during training, the model gives ~84% in both precision and recall which is pretty nice.

# Summary

I really enjoyed working on the implementation. I learned a lot about how PyTorch works and how to use it when building a model from scratch.

Even though this is still a work in progress for me, I will be gradually improving the implementation as much as I can, time permitting. Some things I plan to do

* Improve metrics on testing set

* Experiment with binarization of weights: to decrease model size

The implementation of the model can be found [here](https://github.com/iliailmer/harmonic_network).
