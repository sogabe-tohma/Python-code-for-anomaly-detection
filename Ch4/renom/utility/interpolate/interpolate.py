import numpy as np
import itertools
from distutils.version import StrictVersion
from scipy.interpolate import interp1d


class _Interpolate:
    def __init__(self, x, axis=0, missing=(np.Inf, np.nan)):
        self.x = np.array(x)
        self.axis = axis
        self.missing = list(missing)
        self.transpose_list = self._get_transpose_list(self.x, self.axis)
        self.x_transpose = np.transpose(self.x, self.transpose_list)
        self.args = [range(self.x.shape[i]) for i in range(self.x.ndim) if i != self.axis]

    def _get_transpose_list(self, x, axis):
        transpose_list = list(range(x.ndim))
        transpose_list.remove(axis)
        transpose_list.append(axis)
        return transpose_list

    def _get_nomissing_index(self, x_1d, missing):
        cond = np.array([True for _ in range(len(x_1d))])
        for mis in missing:
            if np.isnan(mis):
                cond = cond & ~np.isnan(x_1d)
            else:
                if StrictVersion(np.version.full_version) < StrictVersion("1.13.0"):
                    cond = cond & ~np.in1d(x_1d, [mis])
                else:
                    cond = cond & ~np.isin(x_1d, [mis])
        return np.argwhere(cond).flatten()

    def _linear_interpolate(self):
        for i in itertools.product(*self.args):
            x_1d = self.x_transpose[i]
            index = self._get_nomissing_index(x_1d, self.missing)
            assert len(index) != 0, "Can't interpolate a empty column."
            assert 0 in index and len(x_1d) - 1 in index, "Can't interpolate without the ends."
            f_interp = interp1d(index, x_1d[index])
            self.x_transpose[i] = f_interp(np.arange(0, self.x.shape[self.axis]))
        x_interp = np.transpose(self.x_transpose, self.transpose_list)
        return x_interp

    def _spline_interpolate(self):
        for i in itertools.product(*self.args):
            x_1d = self.x_transpose[i]
            index = self._get_nomissing_index(x_1d, self.missing)
            assert len(index) != 0, "Can't interpolate a empty column."
            assert 0 in index and len(x_1d) - 1 in index, "Can't interpolate without the ends."
            assert len(index) > 3, "Must have at least 4 entries."
            f_interp = interp1d(index, x_1d[index], kind="cubic")
            self.x_transpose[i] = f_interp(np.arange(0, self.x.shape[self.axis]))
        x_interp = np.transpose(self.x_transpose, self.transpose_list)
        return x_interp

    def _constant_interpolate(self, constant):
        for i in itertools.product(*self.args):
            x_1d = self.x_transpose[i]
            index = self._get_nomissing_index(x_1d, self.missing)
            nindex = np.delete(np.arange(0, self.x.shape[self.axis]), index)
            if len(nindex) > 0:
                x_1d[nindex] = constant
            self.x_transpose[i] = x_1d
        x_interp = np.transpose(self.x_transpose, self.transpose_list)
        return x_interp

    def _nearest_index_interpolate(self):
        for i in itertools.product(*self.args):
            x_1d = self.x_transpose[i]
            index = self._get_nomissing_index(x_1d, self.missing)
            nindex = np.delete(np.arange(0, self.x.shape[self.axis]), index)
            if len(nindex) > 0:
                for j in nindex:
                    x_1d[j] = x_1d[index[np.abs(np.array(index) - j).argmin()]]
            self.x_transpose[i] = x_1d
        x_interp = np.transpose(self.x_transpose, self.transpose_list)
        return x_interp


def interpolate(x, mode, axis=0, missing=(np.nan, np.Inf), constant=None):
    assert mode in ["linear", "spline", "constant", "nearest_index"],\
        "specified mode does not exists in interpolate"
    if mode == "constant":
        assert constant is not None, "constant is not specified."

    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, np.ndarray):
        x = x
    else:
        raise TypeError("interpolate does not support {}".format(type(x)))

    try:
        x = x.astype(float)
    except Exception:
        raise ValueError("'{}' is not supported data type".format(x.dtype))

    if mode == "linear":
        return _Interpolate(x, axis, missing)._linear_interpolate()
    if mode == "spline":
        return _Interpolate(x, axis, missing)._spline_interpolate()
    if mode == "constant":
        return _Interpolate(x, axis, missing)._constant_interpolate(constant)
    if mode == "nearest_index":
        return _Interpolate(x, axis, missing)._nearest_index_interpolate()
