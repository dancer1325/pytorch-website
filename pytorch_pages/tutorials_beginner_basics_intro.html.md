# Learn the Basics — PyTorch Tutorials 2.9.0+cu128 documentation

**URL:** https://pytorch.org/tutorials/beginner/basics/intro.html

---

Rate this Page
★
★
★
★
★
beginner/basics/intro
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Note
Go to the end
to download the full example code.
Learn the Basics
||
Quickstart
||
Tensors
||
Datasets & DataLoaders
||
Transforms
||
Build Model
||
Autograd
||
Optimization
||
Save & Load Model
Learn the Basics
#
Created On: Feb 09, 2021 | Last Updated: Dec 04, 2025 | Last Verified: Nov 05, 2024
Authors:
Suraj Subramanian
,
Seth Juarez
,
Cassie Breviu
,
Dmitry Soshnikov
,
Ari Bornstein
Most machine learning workflows involve working with data, creating models, optimizing model
parameters, and saving the trained models. This tutorial introduces you to a complete ML workflow
implemented in PyTorch, with links to learn more about each of these concepts.
We’ll use the FashionMNIST dataset to train a neural network that predicts if an input image belongs
to one of the following classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker,
Bag, or Ankle boot.
This tutorial assumes a basic familiarity with Python and Deep Learning concepts.
Running the Tutorial Code
#
You can run this tutorial in a couple of ways:
In the cloud
: This is the easiest way to get started! Each section has a “Run in Google Colab” link at the top, which opens an integrated notebook in Google Colab with the code in a fully-hosted environment.
Locally
: This option requires you to setup PyTorch and TorchVision first on your local machine (
installation instructions
). Download the notebook or copy the code into your favorite IDE.
How to Use this Guide
#
If you’re familiar with other deep learning frameworks, check out the
0. Quickstart
first
to quickly familiarize yourself with PyTorch’s API.
If you’re new to deep learning frameworks, head right into the first section of our step-by-step guide:
1. Tensors
.
0.
Quickstart
1.
Tensors
2.
Datasets and DataLoaders
3.
Transforms
4.
Build Model
5.
Automatic Differentiation
6.
Optimization Loop
7.
Save, Load and Use Model
Download
Jupyter
notebook:
intro.ipynb
Download
Python
source
code:
intro.py
Download
zipped:
intro.zip
On this page
PyTorch Libraries
torchao
torchrec
torchft
TorchCodec
torchvision
ExecuTorch
PyTorch on XLA Devices