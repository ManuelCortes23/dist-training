"""run.py:
For more information, please refer to the following link:
https://pytorch.org/tutorials/intermediate/dist_tuto.html
"""

import os
from math import ceil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torchvision import datasets, transforms
from utils import Net, DataPartitioner, average_gradients


def partition_dataset():
    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def run():
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(3):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print(
            "Rank ", dist.get_rank(), ", epoch ", epoch, ": ", epoch_loss / num_batches
        )


def init_process(rank, size, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    run()


if __name__ == "__main__":
    world_size = 5
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
