import itertools
import math
import time

import numpy as np
import rospy
from std_msgs.msg import Int32MultiArray

from main import SnakeExp
from PriorityQueue import PriorityQueue
from MDP_model import MdpModel
repeat = 8
motor_vel_step_up = 100
motor_vel_step_down = -100
possible_actions = list(map(list, itertools.product([motor_vel_step_up, motor_vel_step_down], repeat=repeat)))
mdp_model = MdpModel()
rospy.init_node("bfs", anonymous=True)
snake_robot = SnakeExp()
parents = {}
i_s = [2.163330078125,
       9.349365234375,
       -14.8782348632813,
       0.3448486328125,
       8.03310394287109,
       1.63835084438324,
       0.202493816614151,
       2.6513934135437,
       6.86627912521362,
       2.80508852005005,
       1.35032987594605,
       1.65899133682251
       ]
t_s = [0.95758056640625,
       9.3218994140625,
       -14.1091918945313,
       0.350341796875,
       5.86200189590454
       ]


class State:
    def __init__(self, state, last_a=False, parent=False, g_h=0):
        self.angle_state = state[:4]
        self.string_state = state[4:]
        self.full_state = state
        self.last_action = last_a
        self.parent = parent
        self.g_h_score = g_h


def get_path(last_state):
    p = []
    p.insert(last_state)
    parent = last_state.parent
    while parent:
        p.insert(parent)
        parent = parent.parent
    return p


def distance(a, b):
    return math.sqrt(sum(abs(val1 - val2)**2 for val1, val2 in zip(a, b)))


def heuristic(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a, b))


def best_first_search(initial_state, target_state):
    pq = PriorityQueue()
    pq.insert(initial_state)
    while not pq.isEmpty():
        curr_s = pq.pop()
        if curr_s.last_action:
            print("last action: {}".format(curr_s.last_action))
            snake_robot.command = Int32MultiArray(data=np.asarray(curr_s.last_action).astype(int))
            snake_robot.send_motor_cmd(0.2)
            break
        if distance(curr_s.angle_state, target_state) < 0.1:
            path = get_path(curr_s.angle_state)
            return path
        for a in possible_actions:
            s_tag_arr = mdp_model.find_most_likely_successors(initial_state.full_state, a)
            h = heuristic(curr_s.angle_state, target_state)
            g = distance(curr_s.angle_state, target_state)
            s_tag = State(s_tag_arr, a, s, g+h)
            pq.insert(s_tag)


if __name__ == '__main__':
    s = State(i_s)
    d = distance(s.angle_state, t_s)
    while d > 0.1:
        print("initial state: {}".format(s.full_state))
        path = best_first_search(s, t_s)
        time.sleep(0.2)
        s_arr = snake_robot.current_angles_state
        s = State(list(s_arr + snake_robot.current_strings_state))
        d = distance(s.angle_state, t_s)
        print("final state: {}".format(s))
        print("distance: {}".format(d))



