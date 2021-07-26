---
Title: Structural Identifiability Toolbox
date: 2021-7-25
tags: Maple, Symbolic Computing
---

# Introduction

In this repository, I will describe our recently-released Structural Identifiability Toolbox, a web-based application for assessing parameter identifiability of differential models.

[Click here](https://maple.cloud/app/6509768948056064/) to checkout the application! Read on to learn more.

# Why is it better?

The program is fast, free, and is available in _any_ web-browser, including mobile. We take care of a lot stuff in the background letting the user worry only about their model definition without any technicalities.

# Who is it designed for?

Simply put, if you are working with models based on ordinary differential equations and you wish to know structural identifiability properties of your models' parameters then this app is built for you!

Knowing structural identifiability properties will help one set up better experiments to obtain data for the underlying process and extract parameter values correctly.

# What can it do?

## Individual Parameters

The application can answer questions about local or global identifiability parameters (including initial conditions): you provide the input system, choose the probability of correctness and, optionally, specify which parameter to check (by default it checks for all possible parameters). For this, the app is using SIAN algorithm (see [1][1] and [2][2] for details) which is fast, robust, and is correct with user-specified probability.

## Parameter Combinations

If a parameter is _non_-identifiable, one may wish to seek an identifiable function that contains this parameter (and, possibly, others). Built with the algorithm from [3][3], the app can quickly assess generators for all such functions. Moreover, we provide a way to assess whether 1 or more experiments are required to do so, this is called multi-experiment identifiability.

# References

[1]: H. Hong, A. Ovchinnikov, G. Pogudin, C. Yap, [Global Identifiability of Differential Models](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.21921), Communications on Pure and Applied Mathematics, Volume 73, Issue 9, Pages 1831-1879, 2020

[2]: H. Hong, A. Ovchinnikov, G. Pogudin, C. Yap, [SIAN: Software for Structural Identifiability Analysis of ODE Models](https://doi.org/10.1093/bioinformatics/bty1069) Bioinformatics, Volume 35, Issue 16, Pages 2873–2874, 2019

[3]: [Computing All Identifiable Functions of ODE Models](https://arxiv.org/abs/2004.07774), arXiv:2004.07774