'''rubik.py
'''
# <METADATA>
QUIET_VERSION = "0.1"
PROBLEM_NAME = "Rubik Cube"
PROBLEM_VERSION = "0.1"
PROBLEM_AUTHORS = ['M.Ho', 'G. Vincent']
PROBLEM_CREATION_DATE = "1-MAR-2019"
PROBLEM_DESC = \
    '''This formulation of the Rubik Cube uses generic
    Python 3 constructs and has been tested with Python 3.6.
    '''

import numpy as np
import json
import random
import copy

# </METADATA>

# <COMMON_DATA>
F = 0 #front
B = 1 #back
U = 2 #upper
D = 3 #down
L = 4 #left
R = 5 #right

ORDER = ['Front', 'Back', 'Up', 'Down', 'Left', 'Right']
# </COMMON_DATA>

# <COMMON_CODE>
class State:
    def __init__(self, cube=None):
        if cube == None:
            cube = {}
            cube["front"] = np.ones((2, 2)) * 0
            cube["back"] = np.ones((2, 2)) * 1
            cube["up"] = np.ones((2, 2)) * 2
            cube["down"] = np.ones((2, 2)) * 3
            cube["left"] = np.ones((2, 2)) * 4
            cube["right"] = np.ones((2, 2)) * 5
        self.cube = cube

    def __eq__(self, s2):
        return np.array_equal(self.cube["front"], s2.cube["front"]) and \
               np.array_equal(self.cube["back"], s2.cube["back"]) and \
               np.array_equal(self.cube["up"], s2.cube["up"]) and \
               np.array_equal(self.cube["down"], s2.cube["down"]) and \
               np.array_equal(self.cube["left"], s2.cube["left"]) and \
               np.array_equal(self.cube["right"], s2.cube["right"])

    def __str__(self):
        # Produces a textual description of a state.
        # Might not be needed in normal operation with GUIs.
        return str(self.cube)

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        # Performs an appropriately deep copy of a state,
        # for use by operators in creating new states.
        news = State({})
        news.cube = copy.deepcopy(self.cube)
        return news

    def shuffle_cube(self):
        for i in range(100):
            s = self
            s = OPERATORS[random.randint(0, 5)].apply(s)
        return s


def goal_test(s):
    for key, value in s.cube.items():
        if len(set(value.flatten())) > 1:
            return False
    return True
def goal_message(s):
    return "You Solved the Puzzle!"


class Operator:
    def __init__(self, name, state_transf):
        self.name = name
        self.state_transf = state_transf

    def apply(self, s):
        return self.state_transf(s)


# </COMMON_CODE>

# <INITIAL_STATE>
# Use default, but override if new value supplied
# by the user on the command line.
# try:
#     import sys
#     init_state_string = sys.argv[2]
#     print("Initial state as given on the command line: " + init_state_string)
#     init_state_list = eval(init_state_string)
#     ## TODO Make cube from passed in list
#
# except:
#     state = State()
#     state = state.shuffle_cube()
#     CREATE_INITIAL_STATE = state

# </INITIAL_STATE>

# <OPERATORS>

OPERATORS = []
OPERATORS.append(Operator("Rotate Up", lambda s: up(s)))
OPERATORS.append(Operator("Rotate Down", lambda s: down(s)))
OPERATORS.append(Operator("Rotate Left", lambda s: left(s)))
OPERATORS.append(Operator("Rotate Right", lambda s: right(s)))
OPERATORS.append(Operator("Rotate Front", lambda s: front(s)))
OPERATORS.append(Operator("Rotate Back", lambda s: back(s)))

# </OPERATORS>

# <GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
# </GOAL_TEST>

# <GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
# </GOAL_MESSAGE_FUNCTION>


# OPERATORS
def up(s):
    ns = s.copy()
    ns.cube["up"] = np.rot90(ns.cube["up"], 3)

    front = ns.cube["front"][0]
    left = ns.cube["left"][0]
    right = ns.cube["right"][0]
    back = ns.cube["back"][0]

    ns.cube["front"][0] = right
    ns.cube["left"][0] = front
    ns.cube["right"][0] = back
    ns.cube["back"][0] = left

    return ns


def front(s):
    ns = s.copy()
    ns.cube["front"] = np.rot90(ns.cube["front"], 3)
    up = ns.cube["up"][1]
    down = ns.cube["down"][0]
    right = ns.cube["right"][:, 0]
    left = ns.cube["left"][:, 1]

    ns.cube["up"][1] = left
    ns.cube["down"][0] = right
    ns.cube["right"][:, 0] = up
    ns.cube["left"][:, 1] = down

    return ns


def back(s):
    ns = s.copy()
    ns.cube["back"] = np.rot90(ns.cube["back"], 3)
    up = ns.cube["up"][0]
    down = ns.cube["down"][1]
    right = ns.cube["right"][:, 0]
    left = ns.cube["left"][:, 1]

    ns.cube["up"][0] = right
    ns.cube["down"][1] = left
    ns.cube["right"][:, 0] = down
    ns.cube["left"][:, 1] = up

    return ns


def down(s):
    ns = s.copy()
    ns.cube["down"] = np.rot90(ns.cube["down"], 3)

    front = ns.cube["front"][1]
    left =  ns.cube["left"][1]
    right =  ns.cube["right"][1]
    back = ns.cube["back"][1]

    ns.cube["front"][1] = left
    ns.cube["left"][1] = back
    ns.cube["right"][1] = front
    ns.cube["back"][1] = right

    return ns

def left(s):
    ns = s.copy()
    ns.cube["left"] = np.rot90(ns.cube["left"], 3)

    up = ns.cube["up"][:, 0]
    back = ns.cube["back"][:, 1]
    down = ns.cube["down"][:, 0]
    front = ns.cube["front"][:, 0]

    ns.cube["up"][:, 1] = back
    ns.cube["back"][:, 0] = down
    ns.cube["down"][:, 1] = front
    ns.cube["front"][:, 1] = up

    return ns


def right(s):
    ns = s.copy()
    ns.cube["right"] = np.rot90(ns.cube["right"], 3)

    back = ns.cube["back"][:, 0]
    up = ns.cube["up"][:, 1]
    front = ns.cube["front"][:, 1]
    down = ns.cube["down"][:, 1]

    ns.cube["back"][:, 1] = up
    ns.cube["up"][:, 0] = front
    ns.cube["front"][:, 0] = down
    ns.cube["down"][:, 0] = back

    return ns

# Q-Learning

ACTIONS = [op.name for op in OPERATORS]

# Transition Function, probability of all moves is 1
def T(s, a, sp):
    return 1

# reward function
def R(s, a, sp):
    if goal_test(sp):
        return 10000
    else:
        return -1


class MDP_rubik:
    def __init__(self, T, R, start, actions, operators):
        self.T = T
        self.R = R
        self.start_state = start
        self.ACTIONS = actions
        self.OPERATORS = operators
        self.state_sucessor_dict = {}
        self.all_states = set()
        self.curr_state = start


    def take_action(self, a):
        for operator in self.OPERATORS:
            if operator.name == a:
                return operator.apply(self.curr_state)

    def generate_succesors(self, state):
        if state in self.state_sucessor_dict:
            return self.state_sucessor_dict[state]
        successors = []
        for operator in self.OPERATORS:
            ns = operator.apply(state)
            self.all_states.add(ns)
            successors.append(ns)
        self.state_sucessor_dict[state] = successors
        return successors

    def init_q_learn(self):
        self.generate_all_states()
        self.QValues = {}
        self.visit_count = {}
        for s in self.all_states:
            for a in self.ACTIONS:
                self.QValues[(s, a)] = 0
                self.visit_count[(s, a)] = 0

    def get_best_action(self,s):
        best_action = ""
        max_value = float("-inf")
        for a in self.ACTIONS:
            if self.QValues[(s,a)] > max_value:
                best_action = a
                max_value = self.QValues[(s,a)]
        return best_action

    def choose_action(self, s, learning_bias):
        if random.random() < learning_bias:
            return self.ACTIONS[random.randint(0, 5)]
        return self.get_best_action(s)

    def calculate_learning_rate(self, s, a):
        return 1 / self.visit_count.get(s, a)


    def calculate_Q(self,s ,a ,discount, learning_bias):
        curr_q = self.QValues[(s, a)]
        learning_rate = self.calculate_learning_rate(s, a)
        sp = self.take_action(s)
        if self.T(s, a, sp) == 0:
            self.curr_state = s
            return curr_q
        best_action = self.get_best_action(s)
        new_val = self.R(s, a, sp) + discount * self.QValues[(sp, best_action)]
        biased_val = (1 - learning_rate) * curr_q + learning_rate * new_val
        return biased_val


    def QLearn(self, iterations, discount, learning_bias):
        self.init_q_learn()
        for i in range(iterations):
            self.curr_state = self.start_state
            while not goal_test(self.curr_state):
                s = self.curr_state
                a = self.choose_action(s, learning_bias)
                self.visit_count[(s, a)] += 1
                self.QValues[(s, a)] = self.calculate_Q(s, a, discount, learning_bias)


    def generate_all_states(self):
        OPEN = [self.start_state]
        CLOSED = []

        while OPEN != []:
            S = OPEN[0]
            del OPEN[0]
            CLOSED.append(S)
            L = self.generate_succesors(S)
            for state in L:
                if state not in CLOSED:
                    OPEN.append(state)
            print(len(OPEN))
    def getPolicyDict(self):
        self.opt_policy = {}
        for state in self.all_states:
            self.opt_policy[state] = self.get_best_action(state)

state = State()
state = state.shuffle_cube()
CREATE_INITIAL_STATE = state
mdp = MDP_rubik(T, R, CREATE_INITIAL_STATE, ACTIONS, OPERATORS)
mdp.QLearn(1, .8, .5)
