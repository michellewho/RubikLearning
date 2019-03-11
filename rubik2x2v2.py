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


class My_Priority_Queue:
    def __init__(self):
        self.q = []  # Actual data goes in a list.

    def __contains__(self, elt):
        '''If there is a (state, priority) pair on the list
        where state==elt, then return True.'''
        # print("In My_Priority_Queue.__contains__: elt= ", str(elt))
        for pair in self.q:
            if pair[0] == elt: return True
        return False

    def delete_max(self):
        ''' Standard priority-queue dequeuing method.'''
        if self.q == []: return []  # Simpler than raising an exception.
        temp_min_pair = self.q[0]
        temp_min_value = temp_min_pair[1]
        temp_min_position = 0
        for j in range(1, len(self.q)):
            if self.q[j][1] > temp_min_value:
                temp_min_pair = self.q[j]
                temp_min_value = temp_min_pair[1]
                temp_min_position = j
        del self.q[temp_min_position]
        return temp_min_pair

    def insert(self, state, priority):
        '''We do not keep the list sorted, in this implementation.'''
        # print("calling insert with state, priority: ", state, priority)

        if self[state] != -1:
            print("Error: You're trying to insert an element into a My_Priority_Queue instance,")
            print(" but there is already such an element in the queue.")
            return
        self.q.append((state, priority))

    def __len__(self):
        '''We define length of the priority queue to be the
        length of its list.'''
        return len(self.q)

    def __getitem__(self, state):
        '''This method enables Pythons right-bracket syntax.
        Here, something like  priority_val = my_queue[state]
        becomes possible. Note that the syntax is actually used
        in the insert method above:  self[state] != -1  '''
        for (S, P) in self.q:
            if S == state: return P
        return -1  # This value means not found.

    def __delitem__(self, state):
        '''This method enables Python's del operator to delete
        items from the queue.'''
        # print("In MyPriorityQueue.__delitem__: state is: ", str(state))
        for count, (S, P) in enumerate(self.q):
            if S == state:
                del self.q[count]
                return

    def __str__(self):
        txt = "My_Priority_Queue: ["
        for (s, p) in self.q: txt += '(' + str(s) + ',' + str(p) + ') '
        txt += ']'
        return txt



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

    if s.cube["front"][m,n] == 0 and s.cube["back"][m,n] == 1 and s.cube["up"][m,n] == 2 and\
            s.cube["down"][m,n] == 3 and s.cube["left"][m,n] == 4 and s.cube["right"][m,n] == 5:
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

def check_adjacent(cube):
    n = len(cube)
    val = 0
    for i in range(n):
        for j in range(n):
            # u, ur, r, dr
            if i - 1 in range(n):
                val = val + (0 if len(set([cube[i][j], cube[i - 1][j]])) > 1 else 1)
            # if i - 1 in range(n) and j + 1 in range(n):
            #     val = val + (0 if len(set([cube[i][j], cube[i - 1][j + 1]])) > 1 else 1)
            if j + 1 in range(n):
                val = val + (0 if len(set([cube[i][j], cube[i][j + 1]])) > 1 else 1)
            # if i + 1 in range(n) and j + 1 in range(n):
            #     val = val + (0 if len(set([cube[i][j], cube[i + 1][j + 1]])) > 1 else 1)
    return val



# check number of adjacent pairs of same color
def num_adj_front(s):
    return check_adjacent(s.cube["front"])
def num_adj_back(s):
    return check_adjacent(s.cube["back"])
def num_adj_left(s):
    return check_adjacent(s.cube["left"])
def num_adj__right(s):
    return check_adjacent(s.cube["right"])
def num_adj__up(s):
    return check_adjacent(s.cube["up"])
def num_adj__down(s):
    return check_adjacent(s.cube["down"])

# check number of unique colors
def unique_num_front(s):
    return len(set(s.cube["front"].flatten()))
def unique_num_back(s):
    return len(set(s.cube["back"].flatten()))
def unique_num_left(s):
    return len(set(s.cube["left"].flatten()))
def unique_num_right(s):
    return len(set(s.cube["right"].flatten()))
def unique_num_up(s):
    return len(set(s.cube["up"].flatten()))
def unique_num_down(s):
    return len(set(s.cube["down"].flatten()))

# check number of same colors in corners
def unique_corners_front(s):
    return len(set(s.cube["front"][[0, 0, -1, -1], [0, -1, 0, -1]]))
def unique_corners_back(s):
    return len(set(s.cube["back"][[0, 0, -1, -1], [0, -1, 0, -1]]))
def unique_corners_left(s):
    return len(set(s.cube["left"][[0, 0, -1, -1], [0, -1, 0, -1]]))
def unique_corners_right(s):
    return len(set(s.cube["right"][[0, 0, -1, -1], [0, -1, 0, -1]]))
def unique_corners_up(s):
    return len(set(s.cube["up"][[0, 0, -1, -1], [0, -1, 0, -1]]))
def unique_corners_down(s):
    return len(set(s.cube["down"][[0, 0, -1, -1], [0, -1, 0, -1]]))



def unique_list(s):
    return np.array([unique_num_front(s), unique_num_back(s),unique_num_left(s), unique_num_right(s), unique_num_up(s), unique_num_down(s)])

def adj_list(s):
    return np.array([num_adj_front(s), num_adj_back(s),num_adj_left(s), num_adj__right(s), num_adj__up(s), num_adj__down(s)])

def corner_list(s):
    return np.array([unique_corners_front(s), unique_corners_back(s), unique_corners_left(s), unique_corners_right(s), unique_corners_up(s), unique_corners_down(s)])
# Q-Learning

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

    def generate_succesors(self, s):
        if s in self.state_sucessor_dict:
            return self.state_sucessor_dict[s]
        successors = []
        for operator in self.OPERATORS:
            ns = operator.apply(s)
            successors.append(ns)
        self.state_sucessor_dict[s] = successors
        self.all_states.update(successors)
        return successors

    def init_q_learn(self):
        self.generate_all_states()
        for s in self.all_states:
            for a in self.ACTIONS:
                self.QValues[(s, a)] = 0
                self.visit_count[(s, a)] = 0
        # print(self.QValues)

    def get_best_action(self, s):
        best_action = ""
        max_value = float("-inf")
        for a in self.ACTIONS:
            # if (s, a) in self.QValues:
            if self.QValues[(s, a)] > max_value:
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
        best_action = self.get_best_action(s)
        curr_q = self.QValues[(s, best_action)]
        learning_rate = self.calculate_learning_rate(s, a)
        sp = self.take_action(a)
        if sp not in self.all_states:
            return -10
        self.curr_state = sp
        if goal_test(sp):
            return 1000
        best_action = self.get_best_action(sp)
        # print(sp, best_action)
        new_val = self.R(s, a, sp) + discount * self.QValues[(sp, best_action)]

        biased_val = (1 - learning_rate) * curr_q + learning_rate * new_val
        return biased_val


    def QLearn(self, iterations, discount, learning_bias):
        self.init_q_learn()
        for i in range(iterations):
            print(i)
            count = 0
            self.curr_state = self.start_state
            while not goal_test(self.curr_state) and count < 100:
                # print("working")
                s = self.curr_state
                a = self.choose_action(s, learning_bias)
                self.visit_count[(s, a)] += 1
                self.QValues[(s, a)] = self.calculate_Q(s, a, discount, learning_bias)
                count += 1
            if goal_test(self.curr_state):
                print("QLearn got to goal state")

    def h(self, s):
        if goal_test(s):
            # print("we have seen a goal state")
            return 10000000
        total = 0
        for side in adj_list(s):
            total += math.pow(10, side)
        return total


    def generate_all_states(self):
        TOTAL_COST = 0
        COUNT = 0
        MAX_OPEN_LENGTH = 0
        BACKLINKS = {}
        g = {}
        f = {}
        initial_state = self.start_state
        CLOSED = []
        BACKLINKS[initial_state] = None

        OPEN = My_Priority_Queue()

        g[initial_state] = 0.0
        f[initial_state] = g[initial_state] + self.h(initial_state)
        OPEN.insert(initial_state, f[initial_state])

        while len(OPEN) > 0:
            # print(OPEN)
            (S, P) = OPEN.delete_max()
            # print(self.h(S))
            # print("S:")
            # print(S)
            self.all_states.add(S)
            CLOSED.append(S)
            neighbors = self.generate_succesors(S)
            if goal_test(S):
                print("found goal state")
                self.all_states.add(S)
                print(len(self.all_states))
                return
            COUNT += 1
            new_g = g[S] + 1
            for succ in neighbors:
                if succ not in CLOSED and succ not in OPEN:
                    OPEN.insert(succ, new_g + self.h(succ))
                g[succ] = new_g
            self.all_states.add(S)
        return None  # No more states on OPEN, and no goal reached.

    def getPolicyDict(self):
        self.opt_policy = {}
        for state in self.all_states:
            self.opt_policy[state] = self.get_best_action(state)
        return self.opt_policy


state = State()
CREATE_INITIAL_STATE = state.shuffle_cube(15)
print(str(CREATE_INITIAL_STATE))

mdp = MDP_rubik(T, R, CREATE_INITIAL_STATE, ACTIONS, OPERATORS)
mdp.QLearn(100, .8, .2)
policy_dict = mdp.getPolicyDict()

curr_state = CREATE_INITIAL_STATE
while not goal_test(curr_state):
    print(policy_dict[curr_state])
    for operator in OPERATORS:
        if operator.name == policy_dict[curr_state]:
            curr_state = operator.apply(curr_state)
