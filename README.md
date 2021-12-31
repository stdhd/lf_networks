# lf_networks
Sampling 5 differenct motion priors with the same conditioning frame yields in the following sequences:

![](imgs/Sampling30/imagedraw20_1_0.gif?raw=true)
![](imgs/Sampling30/imagedraw20_1_1.gif?raw=true)
![](imgs/Sampling30/imagedraw20_1_2.gif?raw=true)
![](imgs/Sampling30/imagedraw20_1_3.gif?raw=true)
![](imgs/Sampling30/imagedraw20_1_4.gif?raw=true)

## What is this about?
We implement a conditional normalizing flow to sample videos from the BAIR robot pushing dataset.
Inference is carried out using a U-Net, which calculates the time derivative of our latent variables. Subsequently, a Leapfrog integration step is taken to rollout a new latent representation.
This representation finally decoded to the pixel space.

## How do I install requirements?
Requirements used for this project can be found in requirements.txt
To begin with, we use Pytorch and CUDA acceleration.

## How can I reproduce results?
The main.py file initializes all training and test runs, given the name of the parameter file as --exp argument.

One important hint is that training is executed in two steps. First, the inference network is trained: The model learns to reconstruct videos from a latent state.
In the second step, the conditional normalizing flow is trained. For both training steps, there are distinct configuration files.


