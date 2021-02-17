import numpy as np
import renom as rm

# TODO: Dev cross validator


class CrossValidator():

    def __init__(self):
        pass

    def validate(self, trainer, train_distributor, test_distributor, k=4):
        data_size = len(train_distributor)
        one_data_size = data_size // k
        train_loss_curves = []
        test_loss_curves = []
        validate_result = []

        for i in range(k):
            train_dist, test_dist = train_distributor[i * one_data_size:(i + 1) * one_data_size]
            trainer.train(train_dist, test_dist)
            validate_result.append(trainer.test(test_dist))
            train_loss_curves.append(trainer.train_loss_list)
            test_loss_curves.append(trainer.test_loss_list)

        result = {
            "validation": validate_result,
            "train_loss": train_loss_curves,
            "test_loss": test_loss_curves
        }
        return result
