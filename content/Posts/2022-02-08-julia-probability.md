---
Title: Simulating Textbook Probability Problems in Julia
date: 2022-02-06
status: draft
tags: Julia, Probability, Simulation
---

# Introduction

I am fascinated by the concept of Monte-Carlo simulation: you can, in principle, simulate random events and see how these simulations differ from reality (on a small scale, at least!)

Here I would like to show a couple of simple probability theory textbook problems and my attempt at simulating them in Julia.

I will make sure to include cool pictures!

## Problem 1:

Here is the first problem's setting (from M. Baron's "Porbability and Statistics for Computer Science", 3rd edition, Question 2.5):

>  There are 3 independent computer hardware tests; probabilities of each test being positive are 0.2, 0.3, 0.5. What is the probability that at least one is correct?

### Theoretical Solution

How would we solve this on paper? Probability of "at least" is always 1 minus probability of "neither". That is, 

$$\mathbb{P}(\textrm{at least 1 test = True}) = 1 - \mathbb{P}(\textrm{neither test = True})$$

The latter value is, of course, a product of three probabilities of independent events. If our tests are $A,~B,~C$ with complements (opposites) $A^c,~B^c,~C^c$, then: 

$$\mathbb{P}(\textrm{neither test = True}) = \mathbb{P}(A^c\cap B^c \cap C^c)=\mathbb{P}(A^c)\mathbb{P}(B^c)\mathbb{P}(C^c)$$

Note that $\mathbb{P}(A^c) = 1 - \mathbb{P}(A)$ and same for $B,~C$. Thus