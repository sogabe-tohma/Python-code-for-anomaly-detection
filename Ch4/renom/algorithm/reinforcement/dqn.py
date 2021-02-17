
from __future__ import division
from time import sleep
import copy
from tqdm import tqdm
import numpy as np

import renom as rm
from renom.utility.reinforcement.replaybuffer import ReplayBuffer


class DQN(object):
    """DQN class
    This class provides a reinforcement learning agent including training method.

    Args:
        q_network (Model): Q-Network.
        state_size (tuple, list): The size of state.
        action_pattern (int): The number of action pattern.
        gamma (float): Discount rate.
        buffer_size (float, int): The size of replay buffer.
    """

    def __init__(self, q_network, target_q, state_size, action_pattern, gamma=0.99, buffer_size=1e5):
        self._network = q_network
        self._target_network = target_q
        self._action_size = action_pattern
        self._state_size = state_size if hasattr(state_size, "__getitem__") else [state_size, ]
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._buffer = ReplayBuffer([1, ], self._state_size, buffer_size)

    def action(self, state):
        """This method returns an action according to the given state.
        Args:
            state (ndarray): A state of an environment.

        Returns:
            (int, ndarray): Action.
        """
        self._network.set_models(inference=True)
        shape = [-1, ] + list(self._state_size)
        s = state.reshape(shape)
        return np.argmax(self._network(s).as_ndarray(), axis=1)

    def update(self):
        """This function updates target network."""
        # Check GPU data
        self._target_network.copy_params(self._network)

    def train(self, env, loss_func=rm.ClippedMeanSquaredError(), optimizer=rm.Rmsprop(lr=0.00025, g=0.95),
              epoch=100, batch_size=32, random_step=1000, one_epoch_step=20000, test_step=1000,
              test_env=None, update_period=10000, greedy_step=1000000, min_greedy=0.0, max_greedy=0.9,
              test_greedy=0.95, train_frequency=4):
        """This method executes training of a q-network.
        Training will be done with epsilon-greedy method.

        Args:
            env (function): A function which accepts action as an argument
                and returns prestate, state,  reward and terminal.
            loss_func (Model): Loss function for training q-network.
            optimizer (Optimizer): Optimizer object for training q-network.
            epoch (int): Number of epoch for training.
            batch_size (int): Batch size.
            random_step (int): Number of random step which will be executed before training.
            one_epoch_step (int): Number of step of one epoch.
            test_step (int): Number of test step.
            test_env (function): A environment function for test.
            update_period (int): Period of updating target network.
            greedy_step (int): Number of step
            min_greedy (int): Minimum greedy value
            max_greedy (int): Maximum greedy value
            test_greedy (int): Greedy threshold
            train_frequency (int): For the learning step, training is done at this cycle

        Returns:
            (dict): A dictionary which includes reward list of training and loss list.

        Example:
            >>> import renom as rm
            >>> from renom.algorithm.reinforcement.dqn import DQN
            >>>
            >>> q_network = rm.Sequential([
            ...    rm.Conv2d(32, filter=8, stride=4),
            ...    rm.Relu(),
            ...    rm.Conv2d(64, filter=4, stride=2),
            ...    rm.Relu(),
            ...    rm.Conv2d(64, filter=3, stride=1),
            ...    rm.Relu(),
            ...    rm.Flatten(),
            ...    rm.Dense(512),
            ...    rm.Relu(),
            ...    rm.Dense(action_pattern)
            ... ])
            >>>
            >>> state_size = (4, 84, 84)
            >>> action_pattern = 4
            >>>
            >>> def environment(action):
            ...     prestate = ...
            ...     state = ...
            ...     reward = ...
            ...     terminal = ...
            ...     return prestate, state, reward, terminal
            >>>
            >>> # Instantiation of DQN object
            >>> dqn = DQN(model,
            ...           state_size=state_size,
            ...           action_pattern=action_pattern,
            ...           gamma=0.99,
            ...           buffer_size=buffer_size)
            >>>
            >>> # Training
            >>> train_history = dqn.train(environment,
            ...           loss_func=rm.ClippedMeanSquaredError(clip=(-1, 1)),
            ...           epoch=50,
            ...           random_step=5000,
            ...           one_epoch_step=25000,
            ...           test_step=2500,
            ...           test_env=environment,
            ...           optimizer=rm.Rmsprop(lr=0.00025, g=0.95))
            >>>
            Executing random action for 5000 step...
            epoch 000 avg loss:0.0060 avg reward:0.023: 100%|██████████| 25000/25000 [19:12<00:00, 21.70it/s]
                /// Result
                Average train error: 0.006
                Avg train reward in one epoch: 1.488
                Avg test reward in one epoch: 1.216
                Test reward: 63.000
                Greedy: 0.0225
                Buffer: 29537
                ...
            >>>
            >>> print(train_history["train_reward"])

        """

        # History of Learning
        train_reward_list = []
        test_reward_list = []
        train_error_list = []

        greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step

        if test_env is None:
            test_env = env

        print("Executing random action for %d step..." % random_step)
        for r in range(random_step):
            action = int(np.random.rand() * self._action_size)
            prestate, state, reward, terminal = env(action)
            if prestate is not None:
                self._buffer.store(prestate, np.array(action),
                                   np.array(reward), state, np.array(terminal))

        state = None
        prestate = None
        count = 0
        for e in range(epoch):
            loss = 0
            sum_reward = 0
            train_one_epoch_reward = []
            train_each_epoch_reward = []

            test_one_epoch_reward = []
            test_each_epoch_reward = []

            tq = tqdm(range(one_epoch_step))
            for j in range(one_epoch_step):
                if greedy > np.random.rand() and state is not None:
                    action = np.argmax(np.atleast_2d(self._network(state[None, ...])), axis=1)
                else:
                    action = int(np.random.rand() * self._action_size)
                prestate, state, reward, terminal = env(action)
                greedy += g_step
                greedy = np.clip(greedy, min_greedy, max_greedy)
                sum_reward += reward

                if prestate is not None:
                    self._buffer.store(prestate, np.array(action),
                                       np.array(reward), state, np.array(terminal))
                    train_one_epoch_reward.append(reward)
                else:
                    if len(train_one_epoch_reward) > 0:
                        train_each_epoch_reward.append(np.sum(train_one_epoch_reward))
                    train_one_epoch_reward = []

                if j % train_frequency == 0:
                    # Training
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    self._network.set_models(inference=True)
                    self._target_network.set_models(inference=True)

                    target = self._network(train_prestate).as_ndarray()
                    target.setflags(write=True)

                    # train_state = train_state.reshape(batch_size, *self._state_size)
                    value = self._target_network(train_state).as_ndarray(
                    ) * self._gamma * (~train_terminal[:, None])

                    for i in range(batch_size):
                        a = train_action[i, 0].astype(np.integer)
                        target[i, a] = train_reward[i] + value[i, a]

                    self._network.set_models(inference=False)
                    with self._network.train():
                        z = self._network(train_prestate)
                        ls = loss_func(z, target)
                    ls.grad().update(optimizer)
                    loss += ls.as_ndarray()

                    if count % update_period == 0:
                        self.update()
                        count = 0
                    count += 1

                msg = "epoch {:03d} loss:{:6.4f} sum reward:{:5.3f}".format(
                    e, float(ls.as_ndarray()), sum_reward)
                tq.set_description(msg)
                tq.update(1)

            train_reward_list.append(sum_reward)
            train_error_list.append(float(loss) / (j + 1))

            msg = ("epoch {:03d} avg loss:{:6.4f} avg reward:{:5.3f}".format(
                e, float(loss) / (j + 1), sum_reward / one_epoch_step))
            tq.set_description(msg)
            tq.update(0)
            tq.refresh()
            tq.close()

            # Test
            state = None
            sum_reward = 0
            for j in range(test_step):
                if test_greedy > np.random.rand() and state is not None:
                    action = self.action(state)
                else:
                    action = int(np.random.rand() * self._action_size)
                prestate, state, reward, terminal = test_env(action)

                if prestate is not None:
                    test_one_epoch_reward.append(reward)
                else:
                    if len(test_one_epoch_reward) > 0:
                        test_each_epoch_reward.append(np.sum(test_one_epoch_reward))
                    test_one_epoch_reward = []

                sum_reward += float(reward)
            test_reward_list.append(sum_reward)

            tq.write("    /// Result")
            tq.write("    Average train error: {:5.3f}".format(float(loss) / one_epoch_step))
            tq.write("    Avg train reward in one epoch: {:5.3f}".format(
                np.mean(train_each_epoch_reward)))
            tq.write("    Avg test reward in one epoch: {:5.3f}".format(
                np.mean(test_each_epoch_reward)))
            tq.write("    Test reward: {:5.3f}".format(sum_reward))
            tq.write("    Greedy: {:1.4f}".format(greedy))
            tq.write("    Buffer: {}".format(len(self._buffer)))

            sleep(0.25)  # This is for jupyter notebook representation.

        return {"train_reward": train_reward_list,
                "train_error": train_error_list,
                "test_reward": test_reward_list}
