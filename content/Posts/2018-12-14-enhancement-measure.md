---
Title: 'Image Quality Measure'
date: 2018-12-13
analytics: true
slug: image-quality/
tags: image enhancement,image quality
---
A simple function that can be used to justify image quality and control enhancement.

# Introduction
One difficult thing about image enhancement is to actually measure the level of image quality which is quite a subjective task. On the one hand, each individual can perceive the image quality according to their own tastes and preferences. On the other hand, our visual system is the same for all of us, no matter what tastes you have.

An interesting measure that goes forward to unify the subjectivity of human taste and the visual system based perspective was introduced in [this paper](https://pdfs.semanticscholar.org/5ada/c5932775f089eaace7ebc45a6cba89809134.pdf). Let's try it out!

# Code
Well, the essential idea of this measure is that we focus our attention on the image in a blockwise manner. Splitting the image into blocks and then finding the maximal value of the pixel intensity per each block is fairly straightforward and simple to code.

We split the $M\times N$ image, which in computer memory is represented as a matrix of pixel intensities, into submatrices each of size $n\times n$. This results in $k_1$ blocks along the vertical axis of the image and $k_2$ along the horizontal one. The quality measure can be used

$$\sum_{l=1}^{k_1}\sum_{p=1}^{k_2}\frac{\max(W_{lp})}{\min(W_{lp})}.$$

To avoid division by zero, what I do (and this seems to preserve the measure's main purpose) is the following

$$\sum_{l=1}^{k_1}\sum_{p=1}^{k_2}\frac{\max(W_{lp}+1)}{\min(W_{lp}+1)}.$$

The addition $W_{lp}+1$ implies that we add 1 to all elements in the window $W_{lp}$.

I prefer to scale image to have intensities between 0 and 1, but it really is not important, you can use any range as long as it is ```float``` type. This is because of the nature of the division and $\log$ operators.

And here's the code for this function:
```python
def EME(image, window_width, window_height):
  height, width = image.shape
  sum_ = 0
  k = 0                                    # I just decided not to keep
                                           # track of the blocks
  # window_height/ window_width variables take care of number of blocks
  H = np.int(np.floor(window_height / 2))  # range in height, distance from the
                                           # center of the window
  W = np.int(np.floor(window_width / 2))   # range in width, same as above
  for row in range(0 + H, height - H + 1, window_height):
      for column in range(0 + W, width - W + 1, window_width):
        window = image[row - H:row + H + 1, column - W:column + W + 1]
        I_max = window.max()
        I_min = window.min()
        D = (I_max + 1) / (I_min + 1)
        if D < 0.02: # this is also an underflow precaution
          D = 0.02
        k += 1
        sum_ += 20 * np.log(D)
      return sum_ / k
```
# Examples

Let's try it out on some examples. What I am going to do here is, I will use the ```skimage``` library to obtain the data first.

```py
from skimage.data import camera() # my favourite at this point
```
<figure>
  <img src="{static}/images/enhancement-measure/camera.png?raw=true" alt="original"/>
  <figcaption>My favourite sample image.</figcaption>
</figure>
And then lets use simple histogram equalization for image enhancement. This function will enhance the contrast of the image and will bring some details (but also noise! Don't 100% rely on it!)
```py
from skimage.data import camera
from skimage.restoration import denoise_bilateral, denoise_wavelet

original = camera()
enhanced = equalize_hist(original)
image = rescale(enhanced, 0, 255).astype("uint8")
```
And the result is pretty evident
<figure>
  <img src="{static}/images/enhancement-measure/camera_enh.png?raw=true" alt="comparison"/>
  <figcaption>Quality comparison.</figcaption>
</figure>

Of course, histogram equalization produces artifacts, but visually the image is more detailed than originally which corresponds to increase in the measure, which is what we wanted from it in the first place!

# Question

A question that I have for EME, so far just one, but I feel like it's important.

### How can we make it more resistant to noise?
It does decrease for small amounts of noise:

<figure>
  <img src="{static}/images/enhancement-measure/camera_noise_sigma_1.png?raw=true" alt="comparison"/>
  <figcaption>Adding Gaussian noise to the image (right) with $\sigma=1$ decreases the quality measure.</figcaption>
</figure>

But once noise is pretty strong:

<figure>
  <img src="{static}/images/enhancement-measure/camera_noise_sigma_10.png?raw=true" alt="comparison"/>
  <figcaption>Adding Gaussian noise to the image (right) with $\sigma=10$ increases the quality measure.</figcaption>
</figure>

This is not good, particularly when the enhancement we perform somewhere implicitly brings large noise (such as histogram equalization!) forcing us to believe that the quality improved.

The explanation is, of course, that due to random noise, we (with non-zero probability) increase the maximal and decrease the minimal values of pixels and thus cause the change in the final metric value.

But how to make the metric more noise-robust?..

Well, there are certainly options. Some extensions listed [here](https://ieeexplore.ieee.org/abstract/document/6626251/) certainly could do the job and be noise-resistant. But this is probably for some future experiments, who knows!
