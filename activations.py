import pathlib
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from core import ImageSiren

# set the seed in order to have reproducible experiments.
torch.manual_seed(2)


# define a couple of different initialization functions to compare
# and then inspect what influence they have on the activations of
# the activations of the SIREN network


# ones initializes everything with 1s
# eye is an identity matrix
# default : what torch doing by default.
init_functions = {
    "ones": torch.nn.init.ones_,
    "eye": torch.nn.init.eye_,
    "default": partial(torch.nn.init.kaiming_uniform_, a=5 ** (1/2)),
    "paper": None,
}

for fname, func in init_functions.items():
    path = pathlib.Path.cwd() / "tensorboard_logs" / fname
    writer = SummaryWriter(path)

    # create forward hook that will take the activations of a given layer and
    # log them with TensorBoard.
    def fh(inst, inp, out, number=0):
        layer_name = f"{number}_{inst.__class__.__name__}"
        writer.add_histogram(layer_name, out)

    # instantiate the SIREN and I make sure that I pass the correct initialization function.
    model = ImageSiren(hidden_layers=10, hidden_features=200,
                       first_omega=30, hidden_omega=30, custom_init_function_=func)

    # iterate through all the submodules and i make sure I register the above
    # forward hook.
    for i, layer in enumerate(model.net.modules()):
        if not i:
            continue
        layer.register_forward_hook(partial(fh, number=(i+1)//2))

    # prepaer an input that is going to be uniformly distributed in the interval -1 1.

    # this is exactly the kind of input we will be dealing with while training because
    # we will scale the coordinate grid into the interval -1, 1.

    inp = 2 * (torch.rand(10000, 2) - 0.5)
    writer.add_histogram("0", inp) 
    # run the forward pass and since we registered the  forward hooks with 
    # multiple layers> they'er actually going to be executed 
    res = model(inp)
