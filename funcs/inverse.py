import torch


def inverse(*args, method="schulz", **kwargs):
    """
    Compute the approximated logarithm

    Args:
        method (str):
             'schulz': using Newton Schulz method
             'log': using 1/x = exp(-log(x))
    """
    inverse = {"schulz": inverse_schulz, "log": inverse_log}
    return inverse[method](*args, **kwargs)


def inverse_schulz(x, iterations=12):
    """
    Computes an approximation of the matrix inversion using Newton-Schulz
    iterations

    Source NASA: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19920002505.pdf
    """
    assert len(x.shape) >= 2, "Can't compute inverse on non-matrix"
    assert x.shape[-1] == x.shape[-2], "Must be batches of square matrices"

    eye = torch.eye(x.shape[-1])
    # alpha should be sufficiently small to have convergence
    alpha = 0.002
    inverse = alpha * x.t()

    for _ in range(iterations):
        inverse = inverse @ (2 * eye - x @ inverse)

    return inverse


def inverse_log(x, iterations=8):
    """
    Computes an approximation of the matrix inversion using 1/x = exp(-log(x))
    """
    return (-x.log(iterations=iterations)).exp(iterations=iterations)
