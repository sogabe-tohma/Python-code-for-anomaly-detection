from renom.core import UnaryOp
from renom.operation import reshape


class flatten(UnaryOp):

    def __new__(cls, x):
        N = len(x)
        ret = reshape(x, (N, -1))
        return ret


class Flatten:
    """This function flattens an input tensor.
    It does not affect the batch size.

    Example:
        >>> x = np.random.rand(3, 3, 32, 32)
        >>> x = rm.Variable(x)
        >>> z = rm.flatten(x)
        >>> x.shape
        (3, 3, 32, 32)
        >>> z.shape
        (3, 3072)

        >>> # Use as a instance.
        >>> layer = rm.Flatten()
        >>> z = layer(x)
        >>> z.shape
        (3, 3072)
    """

    def __call__(self, x):
        return flatten(x)
