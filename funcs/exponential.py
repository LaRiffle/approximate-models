def exp(*args, method="limit", **kwargs):
    """
    Compute the approximated exponential

    Args:
        method (str):
             'limit': using a limit approximation
    """
    return {"limit": exp_limit}[method](*args, **kwargs)


def exp_limit(x, iterations=8):
    """Approximates the exponential function using a limit approximation:
    exp(x) = \lim_{n -> infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    iterations. We then compute (1 + x / n) once and square `d` times.

    Args:
        iterations (int): number of iterations for limit approximation

    .. inspired by https://github.com/facebookresearch/CrypTen
    """
    return (1 + x / 2 ** iterations) ** (2 ** iterations)
