import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset

# implement a specific way of initializing a linear layer.
# take weight of the linear layer and modify it in place.
# omega : hyperarameter using for scaling


def paper_init_(weight, is_first=False, omega=1):
    """Initialize the weight of the Linear layer.

    Parameters
    ----------
    weight : torch.Tensor
        The learnable 2d weight matrix.

    is_first : bool
        If True, this Linear layer is the very first one in the network.

    omega : float
        Hyperparameter
    """
    # Given the weight matrix, we extract the input features

    # we initialize the weight with a uniform distribution and the bound
    # will depend on whether we are in the first layer

    in_features = weight.shape[1]

    with torch.no_grad():
        if is_first:
            bound = 1 / in_features
        else:
            bound = np.sqrt(6 / in_features) / omega

        weight.uniform_(-bound, bound)

    # The authors of the paper porpose this initialization strategy becuase
    # It will lead to the activations of the neural network having some really
    # nice properties.

    # implementinga single layer of the SIREN network


class Sinelayer(nn.Module):
    """Linear layer followed by the sine activation.

    Parameters
    ----------
    in_features : int
        Nubmer of input features.

    out_features : int
        Nubmer of output features.

    bias : bool
        If True, the bias is included.

    is_first : bool
        If True, then it represents the first layer of the network. Note that it 
        influences the initialization scheme.

    omega : int
        Hyperparameter. Determines scaling.

    custom_init_function_ : None or callable
        If None, then we are going to use the  `paper_init_` defined above.
        Otherwise, any callable that modifies the `weight` parameter in place.

    Attributes
    ----------
    linear : nn.Linear
        Linear layer.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega=30, custom_init_function_=None,):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # we instantiate the linear layer

        if custom_init_function_ is None:
            paper_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function_(self.linear.weight)
        # initialize the weight matrix of the linear layer
        # the reason why we allow for a custom initialzation function is that
        # we want to inspect how the activations of the neural network behave
        # under different initialization schemes
        # However for traning always use the paper initialization that we defined above

    # when it comes to shapes, the forward method behaves exactly like the linear layer
    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, in_features)`. # of pixels~

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, out_features).

        """
        # in here n_samples is just like number of pixels.
        return torch.sin(self.omega * self.linear(x))
    # input go through the linear layer and sin activation.

# actual neural network.
# we can see all of the hidden layer going to have same number of features.


class ImageSiren(nn.Module):
    """Network composed of SineLayers.

    Parameters
    ----------
    hidden_features : int
        Number of hidden features (each hidden layer the same).

    hidden_layers : int
        Number of hidden layers

    fisrt_omega, hidden_omega : float
        Hyperparameter influencing scaling.

    custom_init_function_ : None or callable
        If None, then we are going to use the `paper_init_` defined abobe.
        Otherwise any callable that modifies the `weight` parameter in place.

    Attributes
    ----------
    net : nn.Sequential
        Sequential collection of `SineLayer` and `nn.Linear` at the end.
    """

    def __init__(self, hidden_features, hidden_layers=1, first_omega=30, hidden_omega=30, custom_init_function_=None,):
        super().__init__()
        # we have two input features representing the coordinates of
        # a given pixel and wil habe single output which will represent the
        # predicted intensity(in this example grayscale.)
        super().__init__()
        in_features = 2
        out_features = 1

        net = []
        net.append(Sinelayer(in_features, hidden_features, is_first=True,
                   custom_init_function_=custom_init_function_, omega=first_omega,))
        # we instantiate an empty list and then we
        # append the first SineLayer to it.

        # we iteratively define all the hidden layers and all of them will
        # have the same nubmer of features.
        for _ in range(hidden_layers):
            net.append(Sinelayer(hidden_features, hidden_features, is_first=False,
                       custom_init_function_=custom_init_function_, omega=hidden_omega,))

        # instantiate the last linear layer and we initialize it accordingly
        final_linear = nn.Linear(hidden_features, out_features)
        if custom_init_function_ is None:
            paper_init_(final_linear.weight,
                        is_first=False, omega=hidden_omega)
        else:
            custom_init_function_(final_linear.weight)

        net.append(final_linear)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples , 2)` representing the 2D pixel coordinates.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, 1)` representing the predicted intensities 
        """
        # The input is going to consist of multiple pixel coordinates
        # And the output is predicted intensity for each of the coordinates
        return self.net(x)

# Siren is actually simle multi-layer perceptron that is using the sine
# function as the activation.

# not implementing the training functionality.
# start with a function that generates a grid of coordinates


def generate_coordinates(n):
    """Generate regular grid of 2D coordinates on [0, n] x [0, n].

    Parameters
    ----------
    n : int
        Number of points per dimension

    Returns
    -------
    coords_abs : np.ndarray
        Array of row and column coordinates of shape `(n ** 2, 2)`.
    """

# assume that the image is a square and the only thing we pass into this function
# is the side length then it will generate a grid of coordinates inside of this image
# and it will return it.

    # use numpy's meshgrid to generate rows and columns.
    # they are just in 2D shapes and then we just flatten both of them
    # and stack them into the final array.
    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")
    coords_abs = np.stack([rows.ravel(), cols.ravle()], axis=-1)

    # So the first column of this array will represetns the row coordinates and
    # the second one will represent the column coordinates.
    return coords_abs


# now the goal is to write a dataset is going to yield the coordinates, the
# intensities and the gradients.

class PixelDataset(Dataset):
    # at construction time we will give this data set a grayscale image.
    # and internally it is going to define multiple attributes.
    """Dataset yielding coordinates, intensitives and (higher) derivatives.

    Paramters
    ---------
    img : np.ndarray
        2D image representing a gracscale image.

    Attributes
    ----------
    size : int
        Heght and width of the square image.

    coords_abs : np.ndarray
        Array of shape `(size ** 2, 2)` representing all coordinates of the `img`.

    grad : np.ndarray
        Array of shape `

    laplace : np.ndarray
        Array of shape `(size, size)` representing the approximate laplace operator. 
    """

    def __init__(self, img):
        if not (img.ndim == 2 and img.shape[0] == img.shape[1]):
            raise ValueError("Only 2D square images are supported. ")

        self.img = img
        self.size = img.shape[0]
        self.coords_abs = generate_coordinates(self.size)

        # take the original image and we apply the Sobel filter in both of the
        # directions.
        # Sobel filter actually approximates the first order drivative and additionally
        # adds some smoothing.

        # Why gradients?
        # Cool feature of the SIREN is that if you take the derivative with respect to
        # theinput of any order you will again end up with SIREN. That means that the
        # higher order derivatives are not going to be zero.
        # Therefore the SIREN is very well suited to represent natural signals.
        # We use this property and superview our network on derivatives of any order.
        # Rather than just the zero order intensities. This is exactly the idea what
        # we are doing here.
        self.grad = np.stack([sobel(img, axis=0), sobel(img, axis=1)], axis=-1)

        # We take the gradient and we compute the norm over both of the directions.
        # this way we again have something that has exactly the same shape as our
        # orginal image. And finally we also used the Laplace filter to approximate
        # the Laplace operator.
        # Note that both the Sobel and the Laplace functions are implemented in scipy.
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(img)

    # We define what the size of our data set is and it's nothingele than the number
    # of pixels. Note that this is slighly different to the official implementation
    # becaus there they would set the length equal to one and they would always yield
    # all the pixels of the given image.

    def __len__(self):
        """Determine the number of samples (pixels)."""

        return self.size ** 2
    # implementing the getitem. we will give it the index of the pixel

    def __getitem__(self, idx):
        """Get all relevant data for a single coordinate,"""
        coords_abs = self.coords_abs[idx]
        # we extract its absolute coordinates, unpack it into the rowand column
        # coordinates.
        r, c = coords_abs
        # these relative coordinates are actually going to be the two input features.
        # to our neural network, In general in mahcin learning it's a good thing to
        # scale your features.
        # if our features are uniformly distributed in range minus one to one then
        # the activations throughout the network are going to be really nice

        coords = 2 * ((coords_abs / self.sizse) - 0.5)

        # As you can see we create a dictionary where each entry represents some
        # information about the coordinates of that pixel. So for exmaple the
        # relative coordinates, the absolute coordinates, the intensity, the gradientm
        # the laplace and so on.
        # Important thing : the relative coordinates will be used as features when it
        # comes to the intensity, gradient and laplace they can ll be used to supervise
        # the network. Note that this is extremely powerful because not only can we
        # supervice on the intensity but we can also supervise on any higher order
        # derivative. As you've just seen the __getitem__ returned a dictionary and
        # 

        return {
            "coords": coords,
            "coords_abs": coords_abs,
            "intensity": self.img[r, c],
            "grad_norm": self.grad_norm[r, c],
            "grad": self.grad[r, c],
            "laplace": self.laplace[r, c],
        }

# now the only thing left is to code gradient related uilities that
# will be using torch's autograd.
# Before start, a quick overview of higher order derivatives in torch.
