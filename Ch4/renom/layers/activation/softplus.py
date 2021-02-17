

import numpy as np
import renom as rm
import renom.cuda as cu


class softplus(rm.core.UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        ret = np.log(1 + np.exp(arg))
        return ret

    @classmethod
    def _oper_gpu(cls, arg):
        ret = rm.core.get_gpu(arg).empty_like_me()
        cu.cusoftplus_forward(rm.core.get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, rm.core.Node):
            dx = 1 / (1 + np.exp(-self.attrs._arg))
            self.attrs._arg._update_diff(context, dy * dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, rm.core.Node):
            dx = rm.core.get_gpu(self.attrs._arg).empty_like_me()
            cu.cusoftplus_backward(rm.core.get_gpu(self.attrs._arg), dx, rm.core.get_gpu(dy))
            self.attrs._arg._update_diff(context, dx, **kwargs)


class Softplus:

    def __call__(self, x):
        return softplus(x)
