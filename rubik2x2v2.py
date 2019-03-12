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
import math
from queue import PriorityQueue
from itertools import count


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
        return np.array_equal(self.cube["front"], s2.cube["front"]) and np.array_equal(self.cube["back"], s2.cube["back"]) and \
               np.array_equal(self.cube["left"], s2.cube["left"]) and np.array_equal(self.cube["right"], s2.cube["right"]) and \
               np.array_equal(self.cube["up"], s2.cube["up"]) and np.array_equal(self.cube["down"], s2.cube["down"])
    def __str__(self):
        # Produces a textual description of a state.
        # Might not be needed in normal operation with GUIs.
        txt = "\n"
        txt += str(self.cube["front"])
        txt += str(self.cube["back"])
        txt += str(self.cube["left"])
        txt += str(self.cube["right"])
        txt += str(self.cube["up"])
        txt += str(self.cube["down"])
        return txt

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        # Performs an appropriately deep copy of a state,
        # for use by operators in creating new states.
        news = State({})
        news.cube = copy.deepcopy(self.cube)
        return news

    def shuffle_cube(self, n):
        s = self
        for i in range(n):
            op = OPERATORS[random.randint(0, 5)]
            s = op.apply(s)
        return s


def goal_test(s):
    return len(set(s.cube["front"].flatten())) == 1 and len(set(s.cube["back"].flatten())) == 1 and \
           len(set(s.cube["left"].flatten())) == 1 and len(set(s.cube["right"].flatten())) == 1 and \
           len(set(s.cube["up"].flatten())) == 1 and len(set(s.cube["down"].flatten())) == 1
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
OPERATORS.append(Operator("Rotate Up", lambda s: up_op(up_op(s))))
OPERATORS.append(Operator("Rotate Down", lambda s: down_op(down_op(s))))
OPERATORS.append(Operator("Rotate Left", lambda s: left_op(left_op(s))))
OPERATORS.append(Operator("Rotate Right", lambda s: right_op(right_op(s))))
OPERATORS.append(Operator("Rotate Front", lambda s: front_op(front_op(s))))
OPERATORS.append(Operator("Rotate Back", lambda s: back_op(back_op(s))))

# </OPERATORS>

# <GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
# </GOAL_TEST>

# <GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
# </GOAL_MESSAGE_FUNCTION>


# OPERATORS
def up_op(s):
    ns = s.copy()
    ns.cube["up"] = np.rot90(ns.cube["up"], 3)

    front = np.copy(ns.cube["front"][0])
    left = np.copy(ns.cube["left"][0])
    right = np.copy(ns.cube["right"][0])
    back = np.copy(ns.cube["back"][0])

    ns.cube["front"][0] = right
    ns.cube["left"][0] = front
    ns.cube["right"][0] = back
    ns.cube["back"][0] = left

    return ns


def front_op(s):
    ns = s.copy()
    ns.cube["front"] = np.rot90(ns.cube["front"], 3)

    up = np.copy(ns.cube["up"][1])
    down = np.copy(ns.cube["down"][0])
    right = np.copy(ns.cube["right"][:, 0])
    left = np.copy(ns.cube["left"][:, 1])

    ns.cube["up"][1] = left
    ns.cube["down"][0] = right
    ns.cube["right"][:, 0] = up
    ns.cube["left"][:, 1] = down

    return ns


def back_op(s):
    ns = s.copy()
    ns.cube["back"] = np.rot90(ns.cube["back"], 3)

    up = np.copy(ns.cube["up"][0])
    down = np.copy(ns.cube["down"][1])
    right = np.copy(ns.cube["right"][:, 0])
    left = np.copy(ns.cube["left"][:, 1])

    ns.cube["up"][0] = right
    ns.cube["down"][1] = left
    ns.cube["right"][:, 0] = down
    ns.cube["left"][:, 1] = up

    return ns


def down_op(s):
    ns = s.copy()
    ns.cube["down"] = np.rot90(ns.cube["down"], 3)

    front = np.copy(ns.cube["front"][1])
    left =  np.copy(ns.cube["left"][1])
    right =  np.copy(ns.cube["right"][1])
    back = np.copy(ns.cube["back"][1])

    ns.cube["front"][1] = left
    ns.cube["left"][1] = back
    ns.cube["right"][1] = front
    ns.cube["back"][1] = right

    return ns

def left_op(s):
    ns = s.copy()
    ns.cube["left"] = np.rot90(ns.cube["left"], 3)

    up = np.copy(ns.cube["up"][:, 0])
    back = np.copy(ns.cube["back"][:, 1])
    down = np.copy(ns.cube["down"][:, 0])
    front = np.copy(ns.cube["front"][:, 0])

    ns.cube["up"][:, 0] = back
    ns.cube["back"][:, 1] = down
    ns.cube["down"][:, 0] = front
    ns.cube["front"][:, 0] = up

    return ns


def right_op(s):
    ns = s.copy()
    ns.cube["right"] = np.rot90(ns.cube["right"], 3)

    back = np.copy(ns.cube["back"][:, 0])
    up = np.copy(ns.cube["up"][:, 1])
    front = np.copy(ns.cube["front"][:, 1])
    down = np.copy(ns.cube["down"][:, 1])

    ns.cube["back"][:, 0] = up
    ns.cube["up"][:, 1] = front
    ns.cube["front"][:, 1] = down
    ns.cube["down"][:, 1] = back

    return ns

# # Features
# # check corners
# def corners(s):
#     for key, value in s.cube.items():
#         face = s.cube[key]
#         if len(set(face[::face.shape[0] - 1, ::face.shape[1] - 1].flatten())) > 1:
#             return 0
#     return 1
#
# # check middle horizontal layer -- not plausible for 2x2?
# def horiz_middle(s):
#     for key, value in s.cube.items():
#         face = s.cube[key]
#         if len(set(face[1].flatten())) > 1:
#             return 0
#     return 1
#
# # check middle vertical layer -- also not plausible for 2x2?
# def vert_middle(s):
#     for key, value in s.cube.items():
#         face = s.cube[key]
#         if len(set(face[:,1].flatten())) > 1:
#             return 0
#     return 1
#
# # check to see if random tile matches middle tile -- also not plausible for 2x2?
# def random_square_match(s):
#     for key, value in s.cube.items():
#         m = 1
#         n = 1
#         face = s.cube[key]
#         while (m == 1 and n == 1):
#             m = random.randint(0, len(face))
#             n = random.randint(0, len(face))
#
#         face = s.cube[key]
#         if face[1, 1] == face[m, n]:
#             return 0
#     return 1
#
# # check to see if random tile on each face is on correct side
# def check_random_tile(s):
#     m = random.randint(0, len(s.cube["front"]) - 1)
#     n = random.randint(0, len(s.cube["front"]) - 1)
#
#     if s.cube["front"][m,n] == 0 and s.cube["back"][m,n] == 1 and s.cube["up"][m,n] == 2 and\
#             s.cube["down"][m,n] == 3 and s.cube["left"][m,n] == 4 and s.cube["right"][m,n] == 5:
#         return 1
#
#     return 0
#
# # check to see if there rightmost vertical on each face is the same
# def check_rightmost(s):
#     for key, value in s.cube.items():
#         face = s.cube[key]
#         if len(set(face[:,-1].flatten())) > 1:
#             return 0
#
#     return 1
#
# # check to see if diagonal of face are all same color
# def check_diag(s):
#     for key, value in s.cube.items():
#         face = s.cube[key]
#         if len(set(face.diagonal().flatten())) > 1:
#             return 0
#
#     return 1
#
# def check_adjacent(cube):
#     n = len(cube)
#     val = 0
#     for i in range(n):
#         for j in range(n):
#             # u, ur, r, dr
#             if i - 1 in range(n):
#                 val = val + (0 if len(set([cube[i][j], cube[i - 1][j]])) > 1 else 1)
#             # if i - 1 in range(n) and j + 1 in range(n):
#             #     val = val + (0 if len(set([cube[i][j], cube[i - 1][j + 1]])) > 1 else 1)
#             if j + 1 in range(n):
#                 val = val + (0 if len(set([cube[i][j], cube[i][j + 1]])) > 1 else 1)
#             # if i + 1 in range(n) and j + 1 in range(n):
#             #     val = val + (0 if len(set([cube[i][j], cube[i + 1][j + 1]])) > 1 else 1)
#     return val
#
#
# # check number of adjacent pairs of same color
# def num_adj_front(s):
#     return check_adjacent(s.cube["front"])
# def num_adj_back(s):
#     return check_adjacent(s.cube["back"])
# def num_adj_left(s):
#     return check_adjacent(s.cube["left"])
# def num_adj__right(s):
#     return check_adjacent(s.cube["right"])
# def num_adj__up(s):
#     return check_adjacent(s.cube["up"])
# def num_adj__down(s):
#     return check_adjacent(s.cube["down"])
#
# # check number of unique colors
# def unique_num_front(s):
#     return len(set(s.cube["front"].flatten()))
# def unique_num_back(s):
#     return len(set(s.cube["back"].flatten()))
# def unique_num_left(s):
#     return len(set(s.cube["left"].flatten()))
# def unique_num_right(s):
#     return len(set(s.cube["right"].flatten()))
# def unique_num_up(s):
#     return len(set(s.cube["up"].flatten()))
# def unique_num_down(s):
#     return len(set(s.cube["down"].flatten()))
#
# # check number of same colors in corners
# def unique_corners_front(s):
#     return len(set(s.cube["front"][[0, 0, -1, -1], [0, -1, 0, -1]]))
# def unique_corners_back(s):
#     return len(set(s.cube["back"][[0, 0, -1, -1], [0, -1, 0, -1]]))
# def unique_corners_left(s):
#     return len(set(s.cube["left"][[0, 0, -1, -1], [0, -1, 0, -1]]))
# def unique_corners_right(s):
#     return len(set(s.cube["right"][[0, 0, -1, -1], [0, -1, 0, -1]]))
# def unique_corners_up(s):
#     return len(set(s.cube["up"][[0, 0, -1, -1], [0, -1, 0, -1]]))
# def unique_corners_down(s):
#     return len(set(s.cube["down"][[0, 0, -1, -1], [0, -1, 0, -1]]))
#
#
#
# def unique_list(s):
#     return np.array([unique_num_front(s), unique_num_back(s),unique_num_left(s), unique_num_right(s), unique_num_up(s), unique_num_down(s)])
#
# def adj_list(s):
#     return np.array([num_adj_front(s), num_adj_back(s),num_adj_left(s), num_adj__right(s), num_adj__up(s), num_adj__down(s)])
#
# def corner_list(s):
#     return np.array([unique_corners_front(s), unique_corners_back(s), unique_corners_left(s), unique_corners_right(s), unique_corners_up(s), unique_corners_down(s)])
# Q-Learning
class MDP_rubik:
    def __init__(self, T, R, start, actions, operators):
        self.T = T
        self.R = R
        self.start_state = start
        self.ACTIONS = actions
        self.OPERATORS = operators
        self.curr_state = start
        self.QValues = {}
        self.visit_count = {}
        self.weights = []
        self.all_states = set()
        self.opt_policy = {}

    def take_action(self, a):
        for op in self.OPERATORS:
            if op.name == a:
                ns = op.apply(self.curr_state)
                self.all_states.add(ns)
                return ns

    def get_weights(self):
        self.weights = [0, 0]

    def get_best_action(self, s):
        best_action = None
        max_value = float("-inf")
        for a in self.ACTIONS:
            if (s, a) in self.QValues:
                if self.QValues[(s, a)] > max_value:
                    best_action = a
                    max_value = self.QValues[(s, a)]
        return best_action

    def choose_action(self, s, learning_bias):
        best_action = None
        if random.random() > learning_bias:
            best_action = self.get_best_action(s)
        if best_action is None:
            best_action = random.choice(ACTIONS)
        return best_action

    def calculate_learning_rate(self, s, a):
        return 1 / self.visit_count[(s, a)]

    # returns number of squares on correct face of cube
    def f1(self, s):
        return len(sum(s.cube["front"] == 0)) + len(sum(s.cube["back"] == 1)) + len(sum(s.cube["up"] == 2)) + \
               len(sum(s.cube["down"] == 3) + len(sum(s.cube["left"] == 4)) + len(sum(s.cube["right"] == 5)))

    # returns number of faces that all have same color
    def f2(self, s):
        count = 0
        count += 1 if len(set(s.cube["front"].flatten())) == 1 else 0
        count += 1 if len(set(s.cube["back"].flatten())) == 1 else 0
        count += 1 if len(set(s.cube["left"].flatten())) == 1 else 0
        count += 1 if len(set(s.cube["right"].flatten())) == 1 else 0
        count += 1 if len(set(s.cube["up"].flatten())) == 1 else 0
        count += 1 if len(set(s.cube["down"].flatten())) == 1 else 0

        return count


    def calculate_Q(self,s ,a ,discount, w0):
        learning_rate = self.calculate_learning_rate(s, a)
        sp = self.take_action(a)
        best_action = self.get_best_action(sp)
        self.QValues[(sp, best_action)] = w0 + self.weights[0] * self.f1(sp) + self.weights[1] * self.f2(sp)
        new_q = w0 + self.weights[0] * self.f1(s) + self.weights[1] * self.f2(s)

        delta = self.R(s, a, sp) + discount * self.QValues[(sp, best_action)] - new_q

        self.update_weights(s, learning_rate, delta, w0)

        return new_q

    def update_weights(self, s, learning_rate, delta, w0):
        self.weights[0] = self.weights[0] + learning_rate * delta * self.f1(s)
        self.weights[1] = self.weights[1] + learning_rate * delta * self.f2(s)

        # do we need to normalize?
        # total = sum(self.weights())
        # self.weights = [(w * 1.0)/total for w in self.weights]

    def QLearn(self, iterations, discount, learning_bias):
        total_goal = 0
        for i in range(iterations):
            print(i)
            count = 0
            self.curr_state = self.start_state
            self.get_weights()
            while not goal_test(self.curr_state) and count < 50:
                s = self.curr_state
                a = self.choose_action(s, learning_bias)
                self.visit_count[(s, a)] = self.visit_count[(s, a)] + 1 if (s, a) in self.visit_count else 1
                self.QValues[(s, a)] = self.calculate_Q(s, a, discount, 1)
                count += 1
            if goal_test(self.curr_state):
                print("QLearn got to goal state")
                total_goal += 1
        print("found goal state n times:", total_goal)


    def getPolicyDict(self):
        for s in self.all_states:
            self.opt_policy[s] = self.get_best_action(s)
        return self.opt_policy


ACTIONS = [op.name for op in OPERATORS]

# Transition Function, probability of all moves is 1
def T(s, a, sp):
    if goal_test(s): return 0
    else: return 1

# reward function
def R(s, a, sp):
    if goal_test(s):
        return 10000
    else:
        return -1


state = State()
CREATE_INITIAL_STATE = state.shuffle_cube(15)
print(str(CREATE_INITIAL_STATE))

print(ACTIONS)
mdp = MDP_rubik(T, R, CREATE_INITIAL_STATE, ACTIONS, OPERATORS)
mdp.QLearn(25, .8, .2)
policy_dict = mdp.getPolicyDict()

# curr_state = CREATE_INITIAL_STATE
# while not goal_test(curr_state):
#     print(policy_dict[curr_state])
#     for operator in OPERATORS:
#         if operator.name == policy_dict[curr_state]:
#             curr_state = operator.apply(curr_state)