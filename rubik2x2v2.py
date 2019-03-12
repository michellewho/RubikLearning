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
import math


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
        txt = ""
        txt += "Front:\n " + str(self.cube["front"]) + "\n"
        txt += "Back:\n " + str(self.cube["back"]) + "\n"
        txt += "Left:\n " + str(self.cube["left"]) + "\n"
        txt += "Right:\n " + str(self.cube["right"]) + "\n"
        txt += "Up:\n " + str(self.cube["up"]) + "\n"
        txt += "Down:\n " + str(self.cube["down"]) + "\n"
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

    # returns the new state after applying a specified action
    def take_action(self, a):
        for op in self.OPERATORS:
            if op.name == a:
                ns = op.apply(self.curr_state)
                self.all_states.add(ns)
                return ns

    # sets weights for SARSA algorithm
    def get_weights(self):
        self.weights = [0, 0]

    # finds the best action for a specified state
    def get_best_action(self, s):
        best_action = None
        max_value = float("-inf")
        for a in self.ACTIONS:
            if (s, a) in self.QValues:
                if self.QValues[(s, a)] > max_value:
                    best_action = a
                    max_value = self.QValues[(s, a)]
        return best_action

    # given a learning bias and state, chooses the next action to take from a specific state
    def choose_action(self, s, learning_bias):
        best_action = None
        if random.random() > self.calculate_learning_rate(s, learning_bias):
            best_action = self.get_best_action(s)
        if best_action is None:
            best_action = random.choice(ACTIONS)
        return best_action

    # returns learning rate
    def calculate_learning_rate(self, s, learning_bias):
        if s in self.visit_count:
            return 1 / self.visit_count[s]
        return learning_bias

    # feature 1
    def f1(self, s):
        total = 0
        score_list = np.array([self.check_adjacent(s.cube["front"]), self.check_adjacent(s.cube["back"]),
                               self.check_adjacent(s.cube["left"]), self.check_adjacent(s.cube["right"]),
                               self.check_adjacent(s.cube["up"]), self.check_adjacent(s.cube["down"])])
        for side in score_list:
            total += math.pow(10, side)
        return total

    # checks for adjacent squares within each face of rubik's cube
    def check_adjacent(self, cube):
        n = len(cube)
        val = 0
        for i in range(n):
            for j in range(n):
                if i - 1 in range(n):
                    val = val + (0 if len(set([cube[i][j], cube[i - 1][j]])) > 1 else 1)
                if j + 1 in range(n):
                    val = val + (0 if len(set([cube[i][j], cube[i][j + 1]])) > 1 else 1)
        return val

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

    # calculates q score based on SARSA algorithm
    def calculate_Q(self,s ,a ,discount, w0, learning_bias):
        learning_rate = self.calculate_learning_rate(s, learning_bias)
        sp = self.take_action(a)
        self.curr_state = sp
        best_action = self.get_best_action(sp)

        self.QValues[(sp, best_action)] = w0 + self.weights[0] * self.f1(sp) + self.weights[1] * self.f2(sp)

        new_q = w0 + self.weights[0] * self.f1(s) + self.weights[1] * self.f2(s)

        delta = self.R(s, a, sp) + discount * self.QValues[(sp, best_action)] - new_q

        self.update_weights(s, learning_rate, delta, w0)

        return new_q

    # updates weights after calculating q score on SARSA
    def update_weights(self, s, learning_rate, delta, w0):
        self.weights[0] = self.weights[0] + learning_rate * delta * self.f1(s)
        self.weights[1] = self.weights[1] + learning_rate * delta * self.f2(s)


        total = sum(self.weights)
        if total != 0:
            self.weights = [(w * 1.0)/total for w in self.weights]

    # applies q learning
    def QLearn(self, iterations, discount, learning_bias):
        total_goal = 0
        for i in range(iterations):
            print(i)
            count = 0
            self.curr_state = self.start_state
            self.all_states.add(self.start_state)
            self.get_weights()
            while not goal_test(self.curr_state) and count < 50:
                s = self.curr_state
                self.visit_count[s] = self.visit_count[s] + 1 if s in self.visit_count else 1
                a = self.choose_action(s, learning_bias)
                self.QValues[(s, a)] = self.calculate_Q(s, a, discount, 1, learning_bias)
                count += 1

            if goal_test(self.curr_state):
                print("QLearn got to goal state")
                total_goal += 1

        print("found goal state n times:", total_goal)

    # defines policy
    def getPolicyDict(self):
        for s in self.all_states:
            self.opt_policy[s] = self.get_best_action(s)
        return self.opt_policy


# actions
ACTIONS = [op.name for op in OPERATORS]

# transition Function, probability of all moves is 1
def T(s, a, sp):
    return 1

# reward function
def R(s, a, sp):
    if goal_test(sp):
        return 10000
    else:
        return -1






state = State()

print("Customize the MDP learning below, or press enter all the way through for defaults")

num_shuff = input("How many times should I shuffle?: ")
MDP_iteration = input("How many iterations of learning?: ")
MDP_discount = input("What is the discount?: ")
MDP_learning_rate = input("What is the learning_rate?: ")

if num_shuff == "":
    num_shuff = 15
else:
    num_shuff = int(num_shuff)

if MDP_iteration == "":
    MDP_iteration = 25
else:
    MDP_iteration = int(MDP_iteration)

if MDP_discount == "":
    MDP_discount = 1
else:
    MDP_discount = float(MDP_discount)

if MDP_learning_rate == "":
    MDP_learning_rate = .2
else:
    MDP_learning_rate = float(MDP_learning_rate)




CREATE_INITIAL_STATE = state.shuffle_cube(num_shuff)
print(str(CREATE_INITIAL_STATE))
print(ACTIONS)
mdp = MDP_rubik(T, R, CREATE_INITIAL_STATE, ACTIONS, OPERATORS)
mdp.QLearn(MDP_iteration, MDP_discount, MDP_learning_rate)
policy_dict = mdp.getPolicyDict()
best_first_action = policy_dict[CREATE_INITIAL_STATE]
print("Policy recommends a best first action of:", best_first_action)
print("Q Value of:", mdp.QValues[(CREATE_INITIAL_STATE, best_first_action)])
