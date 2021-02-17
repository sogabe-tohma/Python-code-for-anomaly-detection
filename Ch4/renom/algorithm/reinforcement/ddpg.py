
from time import sleep
import copy
from tqdm import tqdm
import numpy as np
import renom as rm
from renom.utility.reinforcement.replaybuffer import ReplayBuffer


class DDPG(object):

    def __init__(self, critic, actor, state_size, action_size, ganma=0.99, momentm=0.001, buffer_size=1e5):
        assert isinstance(state_size, (list, tuple))
        assert isinstance(action_size, (list, tuple))
        assert momentm <= 1.0 and momentm >= 0

        self._actor = actor
        self._critic = critic
        self._target_actor = copy.deepcopy(actor)
        self._target_critic = copy.deepcopy(critic)

        self._action_size = list(action_size)
        self._state_size = list(state_size)
        self._buffer_size = buffer_size
        self._ganma = ganma
        self._momentum = momentm
        self._buffer = ReplayBuffer(self._action_size, self._state_size, buffer_size)

    def action(self, state):
        self._actor.set_models(inference=True)
        shape = [-1, ] + self._state_size
        s = state.reshape(shape)
        return np.argmax(self._actor(s).as_ndarray(), axis=1)

    def update(self):
        # Check GPU data
        for ac, target_ac in zip(self._actor.iter_models(), self._target_actor.iter_models()):
            if hasattr(ac, "params") and hasattr(target_ac, "params"):
                for k in ac.params.keys():
                    ac.params[k] = (1 - self._momentum) * ac.params[k] + \
                        self._momentum * target_ac.params[k]

        for cr, target_cr in zip(self._critic.iter_models(), self._target_critic.iter_models()):
            if hasattr(cr, "params") and hasattr(target_cr, "params"):
                for k in cr.params.keys():
                    cr.params[k] = (1 - self._momentum) * cr.params[k] + \
                        self._momentum * target_cr.params[k]

    def train(self, env, loss_func=rm.ClippedMeanSquaredError(), optimizer_critic=rm.Adam(lr=0.0001),
              optimizer_actor=rm.Adam(lr=0.0001),
              episode=100, batch_size=32, random_step=1000, one_episode_step=5000, test_step=1000,
              test_env=None, update_period=10000, greedy_step=1000000, min_greedy=0.1,
              max_greedy=0.9, exploration_rate=1., test_greedy=0.95, callbacks=None):

        greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step

        if test_env is None:
            test_env = env

        print("Execute random action for %d step..." % random_step)
        for r in range(random_step):
            action = np.random.rand(*self._action_size)
            prestate, action, reward, state, terminal = env(action)
            if prestate is not None:
                self._buffer.store(prestate, np.array(action),
                                   np.array(reward), state, np.array(terminal))
        state = None
        prestate = None
        count = 0
        for e in range(episode):
            loss = 0
            tq = tqdm(range(one_episode_step))
            for j in range(one_episode_step):
                action = np.atleast_2d(self.action(state[None, ...])) + \
                    np.random.randn(batch_size, self._action_size) * (1 - greedy) * exploration_rate
                prestate, action, reward, state, terminal = env(action)
                greedy += g_step
                greedy = np.clip(greedy, min_greedy, max_greedy)
                if prestate is not None:
                    self._buffer.store(prestate, np.array(action),
                                       np.array(reward), state, np.array(terminal))

                # Training
                train_prestate, train_action, train_reward, train_state, train_terminal = \
                    self._buffer.get_minibatch(batch_size)

                target = np.zeros((batch_size, self._action_size), dtype=state.dtype)
                for i in range(batch_size):
                    target[i, train_action[i, 0].astype(np.integer)] = train_reward[i]

                self._target_actor.set_models(inference=True)
                self._target_critic.set_models(inference=True)
                action_state_value = self._target_critic(
                    train_state, self._target_actor(train_state))
                target += (action_state_value *
                           self._ganma * (~train_terminal[:, None])).as_ndarray()

                self._actor.set_models(inference=True)
                self._critic.set_models(inference=False)
                with self._critic.train():
                    z = self._critic(train_prestate, self._actor(train_prestate))
                    ls = loss_func(z, target)

                with self._actor.prevent_upadate():
                    ls.grad().update(optimizer_critic)

                self._actor.set_models(inference=True)
                self._critic.set_models(inference=False)
                with self._critic.train():
                    z = self._critic(train_prestate, self._actor(train_prestate))

                with self._actor.prevent_upadate():
                    z.grad(-1.).update(optimizer_actor)

                loss += ls.as_ndarray()
                if count % update_period == 0:
                    self.update()
                    count = 0
                count += 1
                tq.set_description("episode {:03d} loss:{:6.4f}".format(e, float(ls.as_ndarray())))
                tq.update(1)
            tq.set_description("episode {:03d} avg loss:{:6.4f}".format(e, float(loss) / (j + 1)))
            tq.update(0)
            tq.refresh()
            tq.close()

            # Test
            state = None
            sum_reward = 0

            for j in range(test_step):
                if state is not None:
                    action = self.action(state) +\
                        np.random.randn(batch_size, self._action_size) * \
                        (1 - test_step) * exploration_rate
                prestate, action, reward, state, terminal = test_env(action)
                sum_reward += float(reward)

            tq.write("    /// Result")
            tq.write("    Average train error:{:1.6f}".format(float(loss) / one_episode_step))
            tq.write("    Test reward:{}".format(sum_reward))
            tq.write("    Greedy:{:1.4f}".format(greedy))
            tq.write("    Buffer:{}".format(len(self._buffer)))

            if isinstance(callbacks, dict):
                func = callbacks.get("end_episode", False)
                if func:
                    func()

            sleep(0.25)  # This is for jupyter notebook representation.
