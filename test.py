import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
\

# the question is how is the data loder actually going to process this
# dictionary.

# Example : writing minimal data set.

# The length of this data set 



class MyDataset(Dataset):
    def __len__(self):
        return 8
    def __getitem__(self, ix):
        return {
            "a" : ix,
            "b" : [ix, ix],
            "c" : np.array([ix, ix]),
            "d" : torch.tensor([ix,ix]),
        }
# what we are tring to answer here is how exactly the dataloader create a batch.

# We instantiate our dat set.
dataset = MyDataset()

dataloader = DataLoader(dataset, batch_size=3) 

batch = next(iter(dataloader))

print(batch)

#ouptut 
# {'a': tensor([0, 1, 2]), 'b': [tensor([0, 1, 2]), tensor([0, 1, 2])], 'c': tensor([[0, 0],
#         [1, 1],
#         [2, 2]]), 'd': tensor([[0, 0],
#         [1, 1],
#         [2, 2]])}

# Default behavior kind of depends on the type of the value of the dictionary.
# a was originally an integer or a float and as we can see the data loader batched
# it into a one-dimensional tensor.
# b wat initially a list of two elements and here the data loader actually did the
# batching element by element
# Tthe c and d are both torch tensors and they are identical.
# Important thing : default behavior turns numpy arrays or floats into torch tensors
# So now we understand what is going to happen when our get item returns a 
# dictionary But what if I was not happy about this default behavior?
# but there is solution

# Lookgin at the doc string of the data loader.
# there is collate function which is a callable and we could actually use
# it to define any batching strategy whatsoever.


