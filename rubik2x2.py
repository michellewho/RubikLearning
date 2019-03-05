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
            cube.front = np.ones((2, 2)) * 0
            cube.back = np.ones((2, 2)) * 1
            cube.up = np.ones((2, 2)) * 2
            cube.down = np.ones((2, 2)) * 3
            cube.left = np.ones((2, 2)) * 4
            cube.right = np.ones((2, 2)) * 5
        self.cube = cube

    def __eq__(self, s2):
        return cmp(self.b, s2.b) == 0

    def __str__(self):
        # Produces a textual description of a state.
        # Might not be needed in normal operation with GUIs.

        txt = "\n"
        for i in range(6):
            txt += str(ORDER[i]) + ": " + str(self.b[i]) + "\n"
        return txt[:-2] + "]"

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
            s = OPERATORS[random.randint(0, len(OPERATORS))].apply(s)
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
try:
    import sys

    init_state_string = sys.argv[2]
    print("Initial state as given on the command line: " + init_state_string)
    init_state_list = eval(init_state_string)
    ## TODO Make cube from passed in list

except:
    state = State()
    state = state.shuffle_cube()
    CREATE_INITIAL_STATE = state

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
    ns.cube.up = np.rot90(ns.cube.up, 3)

    front = ns.cube.front[0]
    left = ns.cube.left[0]
    right = ns.cube.right[0]
    back = ns.cube.back[0]

    ns.cube.front[0] = right
    ns.cube.left[0] = front
    ns.cube.right[0] = back
    ns.cube.back[0] = left

    return ns


def front(s):
    ns = s.copy()
    ns.cube.front = np.rot90(ns.cube.front, 3)
    up = ns.cube.up[1]
    down = ns.cube.down[0]
    right = ns.cube.right[:, 0]
    left = ns.cube.left[:, 1]

    ns.cube.up[1] = left
    ns.cube.down[0] = right
    ns.cube.right[:, 0] = up
    ns.cube.left[:, 1] = down

    return ns


def back(s):
    ns = s.copy()
    ns.cube.back = np.rot90(ns.cube.back, 3)
    up = ns.cube.up[0]
    down = ns.cube.down[1]
    right = ns.cube.right[:, 0]
    left = ns.cube.left[:, 1]

    ns.cube.up[0] = left
    ns.cube.down[1] = right
    ns.cube.right[:,0] = up
    ns.cube.left[:,1]  = down

    return ns


def down(s):
    ns = s.copy()
    ns.cube.down = np.rot90(ns.cube.down, 3)

    front = ns.cube.front[1]
    left =  ns.cube.left[1]
    right =  ns.cube.right[1]
    back = ns.cube.back[1]

    ns.cube.front[1] = right
    ns.cube.left[1] = front
    ns.cube.right[1] = back
    ns.cube.back[1] = left

    return ns

def left(s):
    ns = s.copy()
    ns.cube.left = np.rot90(ns.cube.left, 3)

    up = ns.cube.up[:,1]
    back = ns.cube.back[:,0]
    down = ns.cube.down[:,1]
    front = ns.cube.front[:,1]

    ns.cube.up[:, 1] = front
    ns.cube.back[:, 0] = up
    ns.cube.down[:, 1] = back
    ns.cube.front[:, 1] = down

    return ns



def right(s):
    ns = s.copy()
    ns.cube.right = np.rot90(ns.cube.right, 3)

    back = ns.cube.back[:, 1]
    up = ns.cube.up[:, 0]
    front = ns.cube.front[:,0]
    down = ns.cube.down[:, 0]

    ns.cube.back[:, 1] = down
    ns.cube.up[:, 0] = back
    ns.cube.front[:, 0] = up
    ns.cube.down[:, 0] = front

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
    def __init__(self, T, R, start, goal, actions, operators):
        self.T = T
        self.R = R
        self.start_state = start
        self.goal_state = goal
        self.ACTIONS = actions
        self.OPERATORS = operators
        self.state_sucessor_dict = {}
        self.all_states = set()
        self.curr_state = start


    def take_action(self, a):
        for operator in self.OPERATORS:
            if operator.name == a:
                self.curr_state = operator.apply(self.curr_state)

    def generate_succesors(self, state):
        if state in self.state_sucessor_dict:
            return self.state_sucessor_dict[state]
        successors = []
        for operator in self.OPERATORS:
            ns = operator.apply(state)
            self.all_states.append(ns)
            successors.append(ns)
        self.state_sucessor_dict[state] = successors



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