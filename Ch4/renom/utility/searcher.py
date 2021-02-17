from itertools import product
from collections import OrderedDict
from abc import ABCMeta
import numpy as np
from future.utils import with_metaclass

try:
    import GPy as gp
except Exception:
    gp = None


class Searcher(with_metaclass(ABCMeta, object)):
    """Base class of searcher.

    Searcher classes searches the hyper parameter that
    yields the lowest value.

    Args:
        parameters (dict): Dictionary which contains the parameter
            name as a key and each parameter space as a value.

    Example:
        >>> import renom as rm
        >>> from renom.utility.searchera import GridSearcher
        >>> params = {
        ...     "p1":[1, 2, 3],
        ...     "p2":[4, 5, 6],
        ... }
        ...
        >>> searcher = GridSearcher(params)
        >>>
        >>> for p in searcher.suggest():
        ...     searcher.set_result(p["p1"] + p["p2"])
        ...
        >>> bests = searcher.best()
        >>> for i in range(len(bests)):
        ... print("{}: parameter {} value {}".format(i+1, bests[i][0], bests[i][1]))
        ...
        1: parameter {'p2': 4, 'p1': 1} value 5
        2: parameter {'p2': 4, 'p1': 2} value 6
        3: parameter {'p2': 5, 'p1': 1} value 6
    """

    max_param_size = int(1e5)

    def __init__(self, parameters):
        self._params = parameters
        self._result = []
        self._searched = []
        self._searched_index = []
        self._param_sizes = []
        self._paramd_dict = OrderedDict()
        self._raw_paramd_dict = OrderedDict()
        self._current_param = None
        size = []
        for k, v in sorted(self._params.items(), key=lambda x: x[0]):
            assert isinstance(v, list)
            self._raw_paramd_dict[k] = v
            self._paramd_dict[k] = [i for i, _ in enumerate(v)]
            self._param_sizes.append(int(np.prod(size)))
            size.append(len(v))
        self._size = np.prod(size)

    def set_result(self, result, params=None):
        """
        Set the result of yielded hyper parameter to searcher object.

        Args:
            result (float): The result of yielded hyper parameter.
            params (dict): The hyper parameter which used in model. If None has given,
                the result is considered as it caused by last yielded hyper parameter.
        """
        self._result.append(result)
        if params is None:
            self._searched.append(self._get_param(self._current_param))
            self._searched_index.append(self._to_index(self._searched[-1]))
        else:
            self._searched.append(self._get_param(params))
            self._searched_index.append(self._to_index(self._searched[-1]))

    def suggest(self, max_iter):
        """
        This method yields next hyper parameter.

        Args:
            max_iter (int): Maximum iteration number of parameter search.

        Yields:
            dict: Dictionary of hyper parameter.
        """
        raise NotImplementedError

    def _get_raw_param(self, param):
        return [self._raw_paramd_dict[k][param[i]] for i, k in enumerate(self._raw_paramd_dict.keys())]

    def _get_param(self, raw_param):
        return [self._raw_paramd_dict[k].index(raw_param[k]) for k in self._raw_paramd_dict.keys()]

    def _to_index(self, param):
        return np.sum([param[i] * j for i, j in enumerate(self._param_sizes)])

    def best(self, num=3):
        """
        Returns the best hyper parameters.
        By default, this method returns the top 3 hyper parameter
        as a result of searching.

        Args:
            num (int): The number of hyper parameters.

        Returns:
            list: A list of dictionary of hyper parameters.
        """
        best_params = [({k: v for k, v in zip(self._raw_paramd_dict.keys(), self._get_raw_param(p[0]))}, p[1])
                       for p in sorted(zip(self._searched, self._result), key=lambda x: x[1])[:num]]
        return best_params

    def __len__(self):
        return self._size


class GridSearcher(Searcher):

    """Grid searcher class.

    This class searches better hyper parameter in the parameter space with grid search.

    Args:
        parameters (dict): Dictionary witch contains the parameter name as
            a key and each parameter space as a value.

    """

    def suggest(self):
        for p in product(*list(self._paramd_dict.values())[::-1]):
            ret = {k: self._raw_paramd_dict[k][v]
                   for k, v in zip(self._paramd_dict.keys(), p[::-1])}
            self._current_param = ret
            yield ret


class RandomSearcher(Searcher):

    """Random searcher class.

    This class randomly searches a parameter of the model which yields the
    lowest loss.

    Args:
        parameters (dict): Dictionary which contains the parameter
            name as a key and each parameter space as a value.
    """

    def suggest(self, max_iter=10):
        for _ in range(min(max_iter, len(self))):
            item = None
            iter = product(*list(self._paramd_dict.values())[::-1])
            numbers = list(range(len(self)))
            for n in self._searched_index:
                numbers.remove(n)
            item_number = np.random.choice(numbers)
            for _ in range(item_number + 1):
                item = next(iter)
            ret = {k: self._raw_paramd_dict[k][v]
                   for k, v in zip(self._paramd_dict.keys(), item[::-1])}
            self._current_param = ret
            yield ret


class BayesSearcher(Searcher):

    """Bayes searcher class.

    This class performs hyper parameter search
    based on bayesian optimization.

    Args:
        parameters (dict): Dictionary which contains the parameter
            name as a key and each parameter space as a value.

    Note:
        This class requires the module GPy [1]_.
        You can install it using pip. ``pip install gpy``

    .. [1] GPy - Gaussian Process framework http://sheffieldml.github.io/GPy/
    """

    def __init__(self, parameters):
        super(BayesSearcher, self).__init__(parameters)
        self._rand_search = RandomSearcher(parameters)
        if len(self) > Searcher.max_param_size:
            raise Exception("Bayes searcher can't handle parameter candidates more than 10000.")

        if gp is None:
            raise Exception("The module GPy is not found. To install it ``pip install gpy``.")

    def suggest(self, max_iter=10, random_iter=3):
        """
        Args:
            max_iter (int): Maximum iteration number of parameter search.
            random_iter (int): Number of random search.

        """
        candidates = np.array(list(product(*list(self._paramd_dict.values())[::-1])))
        for p in self._rand_search.suggest(random_iter):
            self._current_param = self._rand_search._current_param
            yield p

        for _ in range(min(max_iter, len(self)) - random_iter):
            x = np.array(self._searched)
            y = np.array(self._result)[:, None]
            model = gp.models.GPRegression(x, y)
            model.optimize()
            mse, var = model._raw_predict(candidates)
            while True:
                index = np.argmin(self.acquisition_UCB(mse, var))
                item = candidates[index]
                if self._to_index(item[::-1]) in self._searched_index:
                    mse[index] = np.Inf
                else:
                    break
            ret = {k: self._raw_paramd_dict[k][v]
                   for k, v in zip(self._paramd_dict.keys(), item[::-1])}
            self._current_param = ret
            yield ret

    def acquisition_UCB(self, mse, var, k=1.0):
        return mse - k * var
