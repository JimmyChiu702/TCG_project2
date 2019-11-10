#!/usr/bin/env python3

"""
Basic framework for developing 2048 programs in Python

Author: Hung Guei (moporgic)
        Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
        http://www.aigames.nctu.edu.tw
"""

from board import board
from action import action
from weight import weight
from array import array
import random
import sys

import copy

class agent:
    """ base agent """
    
    def __init__(self, options = ""):
        self.info = {}
        options = "name=unknown role=unknown " + options
        for option in options.split():
            data = option.split("=", 1) + [True]
            self.info[data[0]] = data[1]
        return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def open_episode(self, flag = ""):
        return
    
    def close_episode(self, flag = ""):
        return
    
    def take_action(self, state):
        return action()
    
    def check_for_win(self, state):
        return False
    
    def property(self, key):
        return self.info[key] if key in self.info else None
    
    def notify(self, message):
        data = message.split("=", 1) + [True]
        self.info[data[0]] = data[1]
        return
    
    def name(self):
        return self.property("name")
    
    def role(self):
        return self.property("role")


class random_agent(agent):
    """ base agent for agents with random behavior """
    
    def __init__(self, options = ""):
        super().__init__(options)
        seed = self.property("seed")
        if seed is not None:
            random.seed(int(seed))
        return
    
    def choice(self, seq):
        target = random.choice(seq)
        return target
    
    def shuffle(self, seq):
        random.shuffle(seq)
        return


class weight_agent(agent):
    """ base agent for agents with weight tables """
    
    def __init__(self, options = ""):
        super().__init__(options)
        self.net = []
        init = self.property("init")
        if init is not None:
            self.init_weights(init)
        load = self.property("load")
        if load is not None:
            self.load_weights(load)
        return
    
    def __exit__(self, exc_type, exc_value, traceback):
        save = self.property("save")
        if save is not None:
            self.save_weights(save)
        return
    
    def init_weights(self, info):
        print("Init ")
        self.net = [weight(16777216)] * 16
        return
    
    def load_weights(self, path):
        input = open(path, 'rb')
        size = array('L')
        size.fromfile(input, 1)
        size = size[0]
        for i in range(size):
            self.net += [weight()]
            self.net[-1].load(input)
        return
    
    def save_weights(self, path):
        output = open(path, 'wb')
        array('L', [len(self.net)]).tofile(output)
        for w in self.net:
            w.save(output)
        return


class learning_agent(weight_agent):
    """ base agent for agents with a learning rate """
    
    def __init__(self, options = ""):
        super().__init__(options)
        self.alpha = 0.1
        alpha = self.property("alpha")
        if alpha is not None:
            self.alpha = float(alpha)
        self.last_state = None
        self.last_value = 0
        self.isFirst = False
        self.num_tuples = 16
        self.tuples = [
            [0, 4, 8, 12, 13, 9], [1, 5, 9, 13, 14, 10], [1, 5, 9, 10, 6, 2], [2, 6, 10, 3, 7, 11],
            [3, 2, 1, 0, 4, 5], [7, 6, 5, 4, 8, 9], [7, 6, 5, 9, 10, 11], [11, 10, 9, 13, 14, 15],
            [15, 11, 7, 3, 2, 6], [14, 10, 6, 2, 1, 5], [14, 10, 6, 5, 9, 13], [13, 9, 5, 4, 8, 12],
            [12, 13, 14, 15, 11, 10], [8, 9, 10, 11, 7, 6], [8, 9, 10, 6, 5, 4], [4, 5, 6, 2, 1, 0]
        ]
        return

    def open_episode(self, flag = ""):
        self.isFirst = True
    
    def evaluate(self, state):
        v = 0
        for i in range(self.num_tuples):
            v += self.net[i][self.encode(state.state, self.tuples[i])]
        return v

    def update(self, state, target):
        error = target - self.evaluate(state)
        delta = (self.alpha / 16) * error
        for i in range(self.num_tuples):
            self.net[i][self.encode(state.state, self.tuples[i])] += delta
        return

    def encode(self, state, pos):
        return (state[pos[0]]<<0) | (state[pos[1]]<<4) | (state[pos[2]]<<8) | (state[pos[3]]<<12) | (state[pos[4]]<<16) | (state[pos[5]]<<20)

    def take_action(self, before):
        best_v = float('-inf')
        best_a = None
        best_op = None
        for op in range(4):
            after = board(before)
            reward = after.slide(op)
            if reward != -1:
                tmp_v = reward + self.evaluate(after)
                if tmp_v > best_v:
                    best_v = tmp_v
                    best_a = action.slide(op)
                    best_op = op
        if not self.isFirst:
            if best_v != float('-inf'):
                self.update(self.last_state, best_v)
            else:
                self.update(self.last_state, 0)
        self.last_state = board(before)
        self.last_state.slide(best_op)
        self.last_value = self.evaluate(self.last_state)
        self.isFirst = False

        if best_a == None:
            return action()
        else:
            return best_a
        

class rndenv(random_agent):
    """
    random environment
    add a new random tile to an empty cell
    2-tile: 90%
    4-tile: 10%
    """
    
    def __init__(self, options = ""):
        super().__init__("name=random role=environment " + options)
        return
    
    def open_episode(self, flag = ""):
        self.init_tile_bag()
    
    def take_action(self, state):
        if state.last_move == 0:
            empty = [pos for pos, tile in [(i, state.state[i]) for i in [12, 13, 14, 15]] if not tile]
        elif state.last_move == 1:
            empty = [pos for pos, tile in [(i, state.state[i]) for i in [0, 4, 8, 12]] if not tile]
        elif state.last_move == 2:
            empty = [pos for pos, tile in [(i, state.state[i]) for i in [0, 1, 2, 3]] if not tile]
        elif state.last_move == 3:
            empty = [pos for pos, tile in [(i, state.state[i]) for i in [3, 7, 11, 15]] if not tile]
        else:
            empty = [pos for pos, tile in enumerate(state.state) if not tile]
        if empty:
            pos = self.choice(empty)
            if len(self.tile_bag) == 0:
                self.init_tile_bag()
            tile = self.choice(self.tile_bag)
            self.tile_bag.remove(tile)
            return action.place(pos, tile)
        else:
            return action()
    
    def init_tile_bag(self):
        self.tile_bag = [1, 2, 3]
    
    
class player(random_agent):
    """
    dummy player
    select a legal action randomly
    """
    
    def __init__(self, options = ""):
        super().__init__("name=dummy role=player " + options)
        return
    
    def take_action(self, state):
        legal = [op for op in range(4) if board(state).slide(op) != -1]
        if legal:
            op = self.choice(legal)
            return action.slide(op)
        else:
            return action()
    
    
if __name__ == '__main__':
    print('Threes! Demo: agent.py\n')
    pass
