import numpy as np


class GradientClipping(object):
    """
    This class is used to clip gradient.

    The calculation is dones as shown below:

    .. math::
        \\begin{gather}
        \\hat { g } \\leftarrow \\frac { \\partial L }{ \\partial \\omega }  \\\\
        \\text{ if } || \\hat { g } ||_n \\geq {\\it threshold } \\hspace{5pt} { \\bf then } \\\\
        \\hat { g } \\leftarrow \\frac { threshold } { || \\hat { g } ||_n } \\hat { g } \\\\
        \\end{gather}

    - :math:`L`: Loss
    - :math:`\\omega`: weight
    - :math:`n`: norm

    Args:
        threshold(float): If gradient norm is over this threshold, normalization starts.
        norm(int): Norm value.

    Returns:
        total gradient norm.

    Examples:

        >>> from **** import GradientClipping
        >>> grad_clip = GradientClipping(threshold=0.5,norm=2)
        >>>
        >>> grad = loss.grad()
        >>> grad_clip(grad)
        >>>
        >>> grad.update(Sgd(lr=0.01))

    References:
        | Razvan Pascanu, Tomas Mikolov, Yoshua Bengio
        | On the difficulty of training Recurrent Neural Networks
        | https://arxiv.org/abs/1211.5063

    """

    def __init__(self, threshold=0.5, norm=2):
        self.threshold = threshold
        self.norm = norm

    def __call__(self, gradient=None):
        """
        This function clips the gradient if gradient is above threshold.
        This function will appear in Grad class of ReNom in the near future.
        The calculation is dones as shown below:

        .. math::

            \\hat { g } \\leftarrow \\frac { \\partial \\epsilon }{ \\partial \\theta }  \\\\
            \\text{ if } || \\hat { g } || \\geq {\\it threshold } \\hspace{5pt} { \\bf then }\\\\
            \\hat { g } \\leftarrow \\frac { threshold } { || \\hat { g } || } \\hat { g } \\\\

        Args:
            gradient: gradient object
            threshold(float): threshold
            norm(int): norm of gradient

        Examples::
            >>> from **** import GradientClipping
            >>> grad_clip = GradientClipping(threshold=0.5,norm=2)
            >>>
            >>> grad = loss.grad()
            >>> grad_clip(grad)
            >>>
            >>> grad.update(Sgd(lr=0.01))

        """

        threshold = self.threshold
        norm = self.norm

        assert gradient is not None, "insert the gradient of model (model.grad())"

        # setting variables etc.
        auto_updates = gradient._auto_updates
        variables = gradient.variables
        norm = float(norm)
        threshold = float(threshold)

        if norm == float("inf"):
            # h infinity
            total_norm = np.max([np.max(i) for i in np.max(variables.values())])
        else:
            # regular norm
            total_norm = 0
            for i in auto_updates:
                if i.prevent_update:
                    continue
                arr = np.abs(variables[id(i)])**norm
                total_norm += arr.sum()
            total_norm = total_norm ** (1 / norm)

        # process gradient
        if threshold < total_norm:

            for i in variables:
                variables[i] = threshold * variables[i] / (total_norm + 1e-6)

        return total_norm
