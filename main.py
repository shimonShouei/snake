import random

import numpy as np
import itertools
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray

_max_PWM = 256
_max_angle = 30
_N_joints = 4
_N_string = _N_joints * 2

string_step = 1 / 100
repeat = 8
possible_actions = list(map(list, itertools.product([0, 1], repeat=repeat)))


# 0 = release string, 1 = pull the string

class SnakeExp:

    def __init__(self):
        # ROS
        self.motor_pwm_pub = rospy.Publisher("/robot_snake_4/motor_cmd", Int32MultiArray, queue_size=10)
        self.joint_cmd_pub = rospy.Publisher("/robot_snake_10/joint_cmd", Float32MultiArray, queue_size=10)
        rospy.Subscriber("/robot_snake_1/tension_val", Float32MultiArray, self.tension_val_update)
        rospy.Subscriber("/robot_snake_10/joint_val", Float32MultiArray, self.joint_val_update)
        self.current_angles_state = []
        self.current_strings_state = []
        self.initial_angles_state = []
        self.initial_strings_state = []
        self.final_angles_state = []
        self.final_strings_state = []
        self.command = []

    def write_line(self, file):
        file.write(self.initial_angles_state + '$' + self.initial_strings_state + '$' + self.command + '$' + self.final_angles_state + '$' + self.final_strings_state)

    def get_current_state(self):
        pass

    def move(self):
        pass

    def run_experiment(self):
        file = open('output.txt', 'a')
        while True:
            self.initial_angles_state = self.current_angles_state
            self.initial_strings_state = self.current_strings_state
            self.command = random.choice(possible_actions)
            self.move()
            self.final_angles_state = self.current_angles_state
            self.final_strings_state = self.current_strings_state
            self.write_line(file)

    def joint_val_update(self, msg):
        self.current_angles_state = msg.data

    def tension_val_update(self, msg):
        self.current_strings_state = msg.data


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
