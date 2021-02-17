#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
import numpy as np
from operator import __index__
from multiprocessing import Pipe, Process, Array
import ctypes
import tempfile
import zlib
import struct
import threading
import queue


class DecompThread(threading.Thread):
    QUEUE = queue.Queue()

    def run(self):
        while True:
            q = self.QUEUE.get()
            q()

    @classmethod
    def submit(cls, rec):
        ev = threading.Event()

        def run():
            ev.ret = zlib.decompress(rec)
            ev.set()

        cls.QUEUE.put(run)
        return ev


class ReplayBuffer:
    THREAD_STARTED = False
    # todo: adjust buf size
    BUFSIZE = -1  # 100*1024*1024

    def __init__(self, action_space_size, state_size, buffer_size=1e5):
        self._f = tempfile.TemporaryFile()
        self._toc = {}

        self._size = int(buffer_size)

        self._action_space_shape = list(action_space_size)
        self._state_space_shape = list(state_size)

        self._action_space_size = np.prod(action_space_size) if hasattr(
            action_space_size, "__getitem__") else action_space_size

        self._state_size = np.prod(state_size) if hasattr(state_size, "__getitem__") else state_size

        self._size_prestate = self._state_size * np.float32().itemsize
        self._size_action = self._action_space_size * np.float32().itemsize
        self._size_reward = np.float32().itemsize
        self._size_state = self._state_size * np.float32().itemsize
        self._size_terminal = self._state_size * 1

        self._pos_action = 0 + self._size_prestate
        self._pos_reward = self._pos_action + self._size_action
        self._pos_state = self._pos_reward + self._size_reward
        self._pos_terminal = self._pos_state + self._size_state

        if not ReplayBuffer.THREAD_STARTED:
            # todo: adjust number of threads
            for i in range(4):
                d = DecompThread()
                d.daemon = True
                d.start()
            ReplayBuffer.THREAD_STARTED = True
        self._index = 0
        self._full = False

    def store(self, prestate, action, reward, state, terminal):
        self.add(self._index, prestate, action, reward, state, terminal)
        if self._size < self._index:
            self._full = True
        if not self._full:
            self._index += 1

    def add(self, index, prestate, action, reward, state, terminal):
        # todo: reuse buf when overwriting to the same index
        self._f.seek(0, 2)
        start = self._f.tell()
        c = zlib.compressobj(1)
        self._f.write(c.compress(prestate.astype('float32').tobytes()))
        self._f.write(c.compress(action.astype('float32').tobytes()))
        self._f.write(c.compress(struct.pack('f', reward)))
        self._f.write(c.compress(state.astype('float32').tobytes()))
        self._f.write(c.compress(terminal.tobytes()))  # bool
        self._f.write(c.flush())
        end = self._f.tell()
        self._toc[index] = (start, end)

    def _readrec(self, index):
        f, t = self._toc[index]
        self._f.seek(f, 0)
        rec = self._f.read(t - f)
        return rec

    def _unpack(self, buf):
        prestate = np.frombuffer(buf, np.float32, self._state_size, 0)
        action = np.frombuffer(buf, np.float32, self._action_space_size, self._pos_action)
        reward = struct.unpack('f', buf[self._pos_reward:self._pos_reward + self._size_reward])[0]
        state = np.frombuffer(buf, np.float32, self._state_size, self._pos_state)
        terminal = buf[self._pos_terminal]
        return prestate, action, reward, state, terminal

    def get(self, index):
        buf = self._readrec(index)
        buf = zlib.decompress(buf)
        return self._unpack(buf)

    def get_minibatch(self, batch_size=32, shuffle=True):
        if shuffle:
            if len(self) > 1e3:
                perm = []
                while len(perm) < batch_size:
                    perm = (np.random.rand(batch_size * 2) * len(self)).astype(np.int)
                    perm = list(set(perm))[:batch_size]
            else:
                perm = np.random.permutation(len(self))[:batch_size]
        else:
            perm = np.arange(len(self))

        n = len(perm)
        state_shape = [n, ] + self._state_space_shape
        prestates = np.empty(state_shape, dtype=np.float32)
        actions = np.empty((n, self._action_space_size), dtype=np.float32)
        rewards = np.empty(n, dtype=np.float32)
        states = np.empty(state_shape, dtype=np.float32)
        terminals = np.empty(n, dtype=np.bool)

        events = []
        for index in perm:
            buf = self._readrec(index)
            events.append(DecompThread.submit(buf))

        for i, ev in enumerate(events):
            ev.wait()
            prestate, action, reward, state, terminal = self._unpack(ev.ret)
            prestates[i] = prestate.reshape(self._state_space_shape)
            actions[i] = action
            rewards[i] = reward
            states[i] = state.reshape(self._state_space_shape)
            terminals[i] = terminal

        return prestates, actions, rewards, states, terminals

    def __len__(self):
        return self._index if not self._full else self._size


memmap_path = ".replay_buf"
