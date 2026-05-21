import torch

import diisco.utils as utils


def is_psd(mat):
    is_symmetric = bool((mat == mat.T).all())
    is_positive_definite = bool(torch.all(torch.symeig(mat)[0] > 0))
    return is_symmetric and is_positive_definite


def rbf_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    length_scale: float,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Compute the RBF kernel between x1 and x2.
    Based on Pyro's implementation of the RBF kernel.
    :param x1: First input. shape: (n1, d)
    :param x2: Second input. shape: (n2, d)
    :param length_scale: Length scale of the kernel.
    :return: Kernel matrix. shape: (n1, n2)
    """
    x1_scaled = x1 / length_scale
    x2_scaled = x2 / length_scale
    dists = torch.cdist(x1_scaled, x2_scaled, p=2)
    covariance = variance * torch.exp(-0.5 * dists**2)
    covariance = utils.make_psd(covariance)
    return covariance

def rbf_kernel_double(
    t1: torch.Tensor,
    t2: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
    length_scale_1: float,
    length_scale_2: float,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Compute the RBF kernel between (t1 and t2) and (s1 and s2).
    :param t1: First input. shape: (n1, d)
    :param t2: Second input. shape: (n2, d)
    :param length_scale: Length scale of the kernel.
    :return: Kernel matrix. shape: (n1, n2)
    """
    t1_scaled = t1 / length_scale_1
    t2_scaled = t2 / length_scale_1
    s1_scaled = s1 / length_scale_2
    s2_scaled = s2 / length_scale_2
    dists_t = torch.cdist(t1_scaled, t2_scaled, p=2)
    dists_s = torch.cdist(s1_scaled, s2_scaled, p=2)
    covariance = variance * torch.exp(-0.5 * (dists_t**2 + dists_s**2))
    covariance = utils.make_psd(covariance)
    return covariance
