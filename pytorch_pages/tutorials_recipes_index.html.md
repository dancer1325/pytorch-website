# Recipes — PyTorch Tutorials 2.9.0+cu128 documentation

**URL:** https://docs.pytorch.org/tutorials/recipes_index.html

---

Rate this Page
★
★
★
★
★
recipes_index
Run in Google Colab
Colab
Download Notebook
Notebook
View on GitHub
GitHub
Recipes
#
Recipes are bite-sized, actionable examples of
how to use specific PyTorch features, different
from our full-length tutorials.
Defining a Neural Network
Learn how to use PyTorch's torch.nn package to create and define a neural network for the MNIST dataset.
Basics
What is a state_dict in PyTorch
Learn how state_dict objects and Python dictionaries are used in saving or loading models from PyTorch.
Basics
Warmstarting model using parameters from a different model in PyTorch
Learn how warmstarting the training process by partially loading a model or loading a partial model can help your model converge much faster than training from scratch.
Basics
Zeroing out gradients in PyTorch
Learn when you should zero out gradients and how doing so can help increase the accuracy of your model.
Basics
PyTorch Benchmark
Learn how to use PyTorch's benchmark module to measure and compare the performance of your code
Basics
PyTorch Benchmark (quick start)
Learn how to measure snippet run times and collect instructions.
Basics
PyTorch Profiler
Learn how to use PyTorch's profiler to measure operators time and memory consumption
Basics
PyTorch Profiler with Instrumentation and Tracing Technology API (ITT API) support
Learn how to use PyTorch's profiler with Instrumentation and Tracing Technology API (ITT API) to visualize operators labeling in Intel® VTune™ Profiler GUI
Basics
Dynamic Compilation Control with ``torch.compiler.set_stance``
Learn how to use torch.compiler.set_stance
Compiler
Reasoning about Shapes in PyTorch
Learn how to use the meta device to reason about shapes in your model.
Basics
Tips for Loading an nn.Module from a Checkpoint
Learn tips for loading an nn.Module from a checkpoint.
Basics
(beta) Using TORCH_LOGS to observe torch.compile
Learn how to use the torch logging APIs to observe the compilation process.
Basics
Extension points in nn.Module for loading state_dict and tensor subclasses
New extension points in nn.Module.
Basics
torch.export AOTInductor Tutorial for Python runtime
Learn an end-to-end example of how to use AOTInductor for python runtime.
Basics
Demonstration of torch.export flow, common challenges and the solutions to address them
Learn how to export models for popular usecases
Compiler,TorchCompile
Model Interpretability using Captum
Learn how to use Captum attribute the predictions of an image classifier to their corresponding image features and visualize the attribution results.
Interpretability,Captum
How to use TensorBoard with PyTorch
Learn basic usage of TensorBoard with PyTorch, and how to visualize data in TensorBoard UI
Visualization,TensorBoard
Automatic Mixed Precision
Use torch.cuda.amp to reduce runtime and save memory on NVIDIA GPUs.
Model-Optimization
Performance Tuning Guide
Tips for achieving optimal performance.
Model-Optimization
Optimizing CPU Performance on Intel® Xeon® with run_cpu Script
How to use run_cpu script for optimal runtime configurations on Intel® Xeon CPUs.
Model-Optimization
(beta) Utilizing Torch Function modes with torch.compile
Override torch operators with Torch Function modes and torch.compile
Model-Optimization
(beta) Compiling the Optimizer with torch.compile
Speed up the optimizer using torch.compile
Model-Optimization
(beta) Running the compiled optimizer with an LR Scheduler
Speed up training with LRScheduler and torch.compiled optimizer
Model-Optimization
(beta) Explicit horizontal fusion with foreach_map and torch.compile
Horizontally fuse pointwise ops with torch.compile
Model-Optimization
Using User-Defined Triton Kernels with ``torch.compile``
Learn how to use user-defined kernels with ``torch.compile``
Model-Optimization
Compile Time Caching in ``torch.compile``
Learn how to use compile time caching in ``torch.compile``
Model-Optimization
Compile Time Caching Configurations
Learn how to configure compile time caching in ``torch.compile``
Model-Optimization
Reducing torch.compile cold start compilation time with regional compilation
Learn how to use regional compilation to control cold start compile time
Model-Optimization
Intel® Neural Compressor for PyTorch
Ease-of-use quantization for PyTorch with Intel® Neural Compressor.
Quantization,Model-Optimization
Getting Started with DeviceMesh
Learn how to use DeviceMesh
Distributed-Training
Shard Optimizer States with ZeroRedundancyOptimizer
How to use ZeroRedundancyOptimizer to reduce memory consumption.
Distributed-Training
Direct Device-to-Device Communication with TensorPipe RPC
How to use RPC with direct GPU-to-GPU communication.
Distributed-Training
Getting Started with Distributed Checkpoint (DCP)
Learn how to checkpoint distributed models with Distributed Checkpoint package.
Distributed-Training
Asynchronous Checkpointing (DCP)
Learn how to checkpoint distributed models with Distributed Checkpoint package.
Distributed-Training
Getting Started with CommDebugMode
Learn how to use CommDebugMode for DTensors
Distributed-Training
Reducing AoT cold start compilation time with regional compilation
Learn how to use regional compilation to control AoT cold start compile time
Model-Optimization
Edit on GitHub
Show Source
PyTorch Libraries
torchao
torchrec
torchft
TorchCodec
torchvision
ExecuTorch
PyTorch on XLA Devices