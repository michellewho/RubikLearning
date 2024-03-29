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
            cube["front"] = np.ones((3, 3)) * 0
            cube["back"] = np.ones((3, 3)) * 1
            cube["up"] = np.ones((3, 3)) * 2
            cube["down"] = np.ones((3, 3)) * 3
            cube["left"] = np.ones((3, 3)) * 4
            cube["right"] = np.ones((3, 3)) * 5
        self.cube = cube

    def __eq__(self, s2):
        return corners(self) == corners(s2) \
               and check_rightmost(self) == check_rightmost(s2) and check_diag(self) == check_diag(s2) \
                and horiz_middle(self) == horiz_middle(s2) and vert_middle(self) == vert_middle(s2)

    def __str__(self):
        # Produces a textual description of a state.
        # Might not be needed in normal operation with GUIs.
        txt = ""
        txt += str(corners(self))
        # txt += str(check_random_tile(self))
        txt += str(check_rightmost(self))
        txt += str(check_diag(self))
        txt += str(horiz_middle(self))
        txt += str(vert_middle(self))
        return txt

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        # Performs an appropriately deep copy of a state,
        # for use by operators in creating new states.
        news = State({})
        news.cube = copy.deepcopy(self.cube)
        return news

    def shuffle_cube(self):
        s = self
        for i in range(100):
            s = OPERATORS[random.randint(0, 5)].apply(s)
        return s


def goal_test(s):
    return corners(s) == 1 and check_random_tile(s) == 1 and check_rightmost(s) == 1 and check_diag(s) == 1 and \
           horiz_middle(s) == 1 and vert_middle(s) == 1
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

    up = np.copy(ns.cube["up"][2])
    down = np.copy(ns.cube["down"][0])
    right = np.copy(ns.cube["right"][:, 0])
    left = np.copy(ns.cube["left"][:, 2])

    ns.cube["up"][2] = left
    ns.cube["down"][0] = right
    ns.cube["right"][:, 0] = up
    ns.cube["left"][:, 2] = down

    return ns


def back_op(s):
    ns = s.copy()
    ns.cube["back"] = np.rot90(ns.cube["back"], 3)

    up = np.copy(ns.cube["up"][0])
    down = np.copy(ns.cube["down"][2])
    right = np.copy(ns.cube["right"][:, 0])
    left = np.copy(ns.cube["left"][:, 2])

    ns.cube["up"][0] = right
    ns.cube["down"][2] = left
    ns.cube["right"][:, 0] = down
    ns.cube["left"][:, 2] = up

    return ns


def down_op(s):
    ns = s.copy()
    ns.cube["down"] = np.rot90(ns.cube["down"], 3)

    front = np.copy(ns.cube["front"][2])
    left =  np.copy(ns.cube["left"][2])
    right =  np.copy(ns.cube["right"][2])
    back = np.copy(ns.cube["back"][2])

    ns.cube["front"][2] = left
    ns.cube["left"][2] = back
    ns.cube["right"][2] = front
    ns.cube["back"][2] = right

    return ns

def left_op(s):
    ns = s.copy()
    ns.cube["left"] = np.rot90(ns.cube["left"], 3)

    up = np.copy(ns.cube["up"][:, 0])
    back = np.copy(ns.cube["back"][:, 2])
    down = np.copy(ns.cube["down"][:, 0])
    front = np.copy(ns.cube["front"][:, 0])

    ns.cube["up"][:, 0] = back
    ns.cube["back"][:, 2] = down
    ns.cube["down"][:, 0] = front
    ns.cube["front"][:, 0] = up

    return ns


def right_op(s):
    ns = s.copy()
    ns.cube["right"] = np.rot90(ns.cube["right"], 3)

    back = np.copy(ns.cube["back"][:, 0])
    up = np.copy(ns.cube["up"][:, 2])
    front = np.copy(ns.cube["front"][:, 2])
    down = np.copy(ns.cube["down"][:, 2])

    ns.cube["back"][:, 0] = up
    ns.cube["up"][:, 2] = front
    ns.cube["front"][:, 2] = down
    ns.cube["down"][:, 2] = back

    return ns

# Features
# check corners
def corners(s):
    for key, value in s.cube.items():
        face = s.cube[key]
        if len(set(face[::face.shape[0] - 1, ::face.shape[1] - 1].flatten())) > 1:
            return 0
    return 1

# check middle horizontal layer -- not plausible for 2x2?
def horiz_middle(s):
    for key, value in s.cube.items():
        face = s.cube[key]
        if len(set(face[1].flatten())) > 1:
            return 0
    return 1

# check middle vertical layer -- also not plausible for 2x2?
def vert_middle(s):
    for key, value in s.cube.items():
        face = s.cube[key]
        if len(set(face[:,1].flatten())) > 1:
            return 0
    return 1

# check to see if random tile matches middle tile -- also not plausible for 2x2?
def random_square_match(s):
    for key, value in s.cube.items():
        m = 1
        n = 1
        face = s.cube[key]
        while (m == 1 and n == 1):
            m = random.randint(0, len(face))
            n = random.randint(0, len(face))

        face = s.cube[key]
        if face[1, 1] == face[m, n]:
            return 0
    return 1

# check to see if random tile on each face is on correct side
def check_random_tile(s):
    m = random.randint(0, len(s.cube["front"]) - 1)
    n = random.randint(0, len(s.cube["front"]) - 1)

    if s.cube["front"][m,n] == 0 and s.cube["back"][m,n] == 1 and s.cube["up"][m,n] == 0 and\
            s.cube["down"][m,n] == 1 and s.cube["left"][m,n] == 0 and s.cube["right"][m,n] == 1:
        return 1

    return 0

# check to see if there rightmost vertical on each face is the same
def check_rightmost(s):
    for key, value in s.cube.items():
        face = s.cube[key]
        if len(set(face[:,-1].flatten())) > 1:
            return 0

    return 1

# check to see if diagonal of face are all same color
def check_diag(s):
    for key, value in s.cube.items():
        face = s.cube[key]
        if len(set(face.diagonal().flatten())) > 1:
            return 0

    return 1

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
        self.QValues = {}
        self.visit_count = {}

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
                max_value = self.QValues[(s, a)]
        return best_action

    def choose_action(self, s, learning_bias):
        if random.random() < learning_bias:
            return self.ACTIONS[random.randint(0, 5)]
        return self.get_best_action(s)

    def calculate_learning_rate(self, s, a):
        return 1 / self.visit_count[(s, a)]


    def calculate_Q(self,s ,a ,discount, learning_bias):
        curr_q = self.QValues[(s, a)]
        learning_rate = self.calculate_learning_rate(s, a)
        sp = self.take_action(a)
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
            for s in L:
                if s not in CLOSED:
                    print("appending to OPEN")
                    OPEN.append(s)
        print(len(self.all_states))
    def getPolicyDict(self):
        self.opt_policy = {}
        for state in self.all_states:
            self.opt_policy[state] = self.get_best_action(state)


state = State()
print(str(state))
CREATE_INITIAL_STATE = state.shuffle_cube()
print(str(CREATE_INITIAL_STATE))
mdp = MDP_rubik(T, R, CREATE_INITIAL_STATE, ACTIONS, OPERATORS)
mdp.QLearn(1, .8, .5)
