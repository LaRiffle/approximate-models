from funcs import exp


def log(*args, method="householder", **kwargs):
    """
    Compute the approximated logarithm

    Args:
        method (str):
             'householder': using Households method
    """
    log = {"newton": log_newton, "householder": log_householder}
    return log[method](*args, **kwargs)


def log_newton(x, iterations=2, exp_iterations=8):
    """Approximates the logarithm using the Newton Raphson method

    Args:
        iterations (int): number of iterations for Newton Raphson approximation.
        exp_iterations (int): number of iterations for limit approximation of exp

    .. inspired by https://github.com/facebookresearch/CrypTen
    """
    y = x / 40 + 1.9 - 8 * exp(-2 * x - 0.3, iterations=exp_iterations)

    for i in range(iterations):
        h = [1 - x * exp((-y).refresh(), iterations=exp_iterations)]
        for i in range(1, 5):
            h.append(h[-1] * h[0])

        y -= h[0] * (1 + h[0] + h[1] + h[2] + h[3] + h[4])

    return y


def log_householder(x, iterations=2, exp_iterations=8):
    """Approximates the logarithm using 6th order modified Householder iterations.
    Recall that Householder method is an algorithm to solve a non linear equation f(x) = 0.
    Here  f: x -> 1 - C * exp(-x)  with C = self

    Iterations are computed by:
        y_0 = some constant
        h = 1 - self * exp(-y_n)
        y_{n+1} = y_n - h * (1 + h / 2 + h^2 / 3 + h^3 / 6 + h^4 / 5 + h^5 / 7)

    Args:
        iterations (int): number of iterations for 6th order modified
            Householder approximation.
        exp_iterations (int): number of iterations for limit approximation of exp

    .. inspired by https://github.com/facebookresearch/CrypTen
    """

    # Initialization to a decent estimate (found by qualitative inspection):

    y = x / 31 + 1.59 - 20 * exp(-2 * x - 1.47, iterations=exp_iterations)

    # 6th order Householder iterations
    for i in range(iterations):
        h = [1 - x * exp((-y).refresh(), iterations=exp_iterations)]
        for i in range(1, 5):
            h.append(h[-1] * h[0])

        y -= h[0] * (1 + h[0] / 2 + h[1] / 3 + h[2] / 4 + h[3] / 5 + h[4] / 6)

    return y
