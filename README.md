# Distributed Training with PyTorch

This repository demonstrates the progression from single-GPU to multi-GPU distributed training using PyTorch. The scripts illustrate different approaches to parallelize training across multiple GPUs.

## Scripts Overview

1. **single_gpu.py**
   - **Purpose**: Basic training on a single GPU.
   - **Core Structure**: 
     - Utilizes a `Trainer` class to manage training.
     - Loads data using `DataLoader` with shuffling.
     - Saves model checkpoints periodically.

2. **multigpu.py**
   - **Purpose**: Multi-GPU training using PyTorch's `DistributedDataParallel` (DDP).
   - **Core Structure**:
     - Uses `torch.multiprocessing.spawn` to launch processes for each GPU.
     - Initializes DDP with `init_process_group`.
     - Employs `DistributedSampler` to partition data across GPUs.
     - Wraps the model with DDP for gradient synchronization.
     - Saves checkpoints only from the primary process.

3. **multigpu_torchrun.py**
   - **Purpose**: Multi-GPU training using `torchrun` for easier process management.
   - **Core Structure**:
     - Simplifies setup by leveraging `torchrun` to handle process spawning.
     - Uses environment variables for GPU assignment.
     - Similar to `multigpu.py` but with a focus on ease of use and snapshot saving for resuming training.

4. **multinode.py**
   - **Purpose**: Multi-node, multi-GPU training using `torchrun` for distributed training across multiple machines.
   - **Core Structure**:
     - Uses `torchrun` to manage processes across nodes.
     - Sets up DDP with `init_process_group` using environment variables for node and rank management.
     - Similar to `multigpu_torchrun.py` but extends to multiple nodes.
     - Saves snapshots for resuming training, ensuring only one process per node handles saving.

## Key Differences

- **Process Management**: 
  - `single_gpu.py` runs on a single process.
  - `multigpu.py` uses `torch.multiprocessing.spawn` for process management.
  - `multigpu_torchrun.py` and `multinode.py` use `torchrun`, eliminating the need for manual process spawning.

- **Data Handling**:
  - `single_gpu.py` uses standard `DataLoader` with shuffling.
  - `multigpu.py`, `multigpu_torchrun.py`, and `multinode.py` use `DistributedSampler` to ensure each GPU processes a unique subset of data.

- **Model Wrapping**:
  - `single_gpu.py` directly uses the model.
  - `multigpu.py`, `multigpu_torchrun.py`, and `multinode.py` wrap the model with `DistributedDataParallel` for synchronized gradient updates.

- **Node Management**:
  - `multinode.py` extends the multi-GPU setup to multiple nodes, requiring network configuration and rendezvous settings.
  - Utilizes `torchrun` with additional parameters for node rank and rendezvous endpoint.

- **Checkpointing**:
  - All scripts save model checkpoints, but multi-GPU and multi-node scripts ensure only one process handles saving to avoid conflicts.

This progression from single to multi-GPU and multi-node training highlights the scalability of PyTorch's distributed training capabilities.
