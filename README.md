# NeuralFractal
A Library for Visual Exploration of Dynamical Systems defined by Neural Networks

[Home Page](https://amirabbasasadi.github.io/neural-fractal/)  
![A fractal Generated by Neural Fractal](https://user-images.githubusercontent.com/8543469/148009825-7c22991e-e33f-4efb-be6f-69db5fc15001.png)  
The dynamical system that generated the above fractal has been explained in the documentation.  

## Features
- Define Dynamical Systems Using Complex-Valued Neural Networks
- GPU Support for Accelerated Sampling and Rendering
- Pseudo-Coloring Utilities
- Built on Top of PyTorch

## About the package
NeuralFractal has been developed for exploring the properties of dynamical systems defined by neural networks and specially complex-valued neural networks. Fractals are visualizations of Chaos. They have infinite self-similar patterns. One way to generate fractals is by applying a function repeatedly on a set of points and keeping the points that do not diverge to infinity. Interestingly, Even dynamical systems constructed by simple functions in this way can generate amazing fractals. But what happens if instead of a simple function we use a neural network? Repeatedly applying a neural network is equivalent to a recurrent neural network which is able to model complicated non-linear dynamical systems. Reservoir computing has demonstrated even completely random RNNs can construct strange and interesting dynamics. This package is an attempt to explore the strange and beautiful world of fractals

## Installation
```
pip install nfractal
```

## Quick Start
See the [Home Page](https://amirabbasasadi.github.io/neural-fractal/)


## Features Under Development
- Automatic pseudo-coloring
- Gnerating zoom animations

## Main Contributors
- Amirabbas Asadi, Independet AI and computer science researcher
