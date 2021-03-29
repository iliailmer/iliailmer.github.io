---
Title: "How to write a decent training loop with enough flexibility."
collection: posts
analytics: true
slug: how-to-make-training-loop 
date: 2019-06-15
tags: deep learning
---

In this post, I briefly describe my experience in setting up training with PyTorch.

## Introduction

PyTorch is an extremely useful and convenient framework for deep learning. When it comes to working on  a deep learning project, I am more comfortable with PyTorch rather than TensorFlow.

In this quick post, I would like to show how one can go about building a custom training loop, something that I struggled when I was getting started. It is a useful skill to be able to build the training loop on your own because that can help you understand better what happens under the hood of a deep learning package that abstracts a lot of nuts and bolts away from the end-user.

## The Overview of Training

When one trains a network, we need to follow a certain paradigm.

First, set the model into training mode.

Second, start iterating through the training set.

For every batch we must:

* compute the output of the network
* compute the loss
* get gradients
* start descending using the optimizer

This last step we acknowledge that the method for optimization of our model is based on a gradient descent. It can be eqither Adam, SGD, or any other (RAdam is the brand new one which seems to beat state of the art).

In code, we can put it in the form like this:

```python
def train(epoch):
  model.train()  # preparing model for training
  for batch in training_set:
    x, y = batch  # unpack the batch
    # the step below is necessary so that we update the gradient only pertinent to the current batch
    optimizer.zero_grad()
    # compute the output
    output = model(x.cuda())
    # calculate the loss function
    loss = criterion(output, y.cuda())
    # calculate the gradient using backpropagation
    loss.backward()
    # take a step with the optimizer
    optimizer.step()

```

### A Trick for Better Training with Lower Memory

A small batch can result in a small gradient. This, in turn, leads to a problem called [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem): the value is so small, computer simple treats it as zero (underflow). To avoid it, a trick of accumulating gradient as you iterate through the dataset. I saw a practical implementation in this [discussion](https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614#latest-662360).

```python
def train_accumulate(epoch, accumulation_step = 1):
  model.eval()  # preparing model for training
  for idx, batch in enumerate(training_set):
    x, y = batch  # unpack the batch
    # compute the output
    output = model(x.cuda())
    # calculate the loss function
    loss = criterion(output, y.cuda())
    # calculate the gradient using backpropagation
    loss.backward()
    if idx%accumulation_step==0:
      # take a step with the optimizer once
      # we accumulated enough gradients
      optimizer.step()
      optimizer.zero_grad()
```

## In Closing: abstracting training loop

In this post I summarized my experience in building a training loop for PyTorch. Lately, I have been using a more abstracted way of training through [Catalyst](https://catalyst-team.github.io/catalyst/). It is a great tool for higher level abstraction during training and a lot of hardwork has been done to take away the hard part of training.

Nevertheless, both, I believe, are equally important: the abstract and the explicit methods.

Thanks for reading!
