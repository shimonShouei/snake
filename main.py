import random
import time
import signal
import sys

import numpy as np
import itertools
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray

_max_PWM = 60
_max_angle = 30
_N_joints = 4
_N_string = _N_joints * 2
max_tension = 14
min_tension = 0.3
motor_vel_step_up = 3
motor_vel_step_down = -3
repeat = 8
possible_actions = list(map(list, itertools.product([motor_vel_step_up, motor_vel_step_down], repeat=repeat)))


# 0 = release string, 1 = pull the string

class SnakeExp:

    def __init__(self):
        # ROS
        self.motor_pwm_pub = rospy.Publisher("/robot_snake_1/motor_cmd", Int32MultiArray, queue_size=10)
        rospy.Subscriber("/robot_snake_1/tension_val", Float32MultiArray, self.tension_val_update)
        rospy.Subscriber("/robot_snake_1/joint_val", Float32MultiArray, self.joint_val_update)
        self.current_angles_state = []
        self.current_strings_state = []
        self.initial_angles_state = []
        self.initial_strings_state = []
        self.final_angles_state = []
        self.final_strings_state = []
        self.command = []

    def write_line(self, file):
        file.write(
            self.initial_angles_state.__str__() + '$' + self.initial_strings_state.__str__() + '$' + self.command.__str__() + '$' + self.final_angles_state.__str__() + '$' + self.final_strings_state.__str__())

    def get_current_state(self):
        pass

    def run_experiment(self):
        file = open('output.txt', 'a')
        while not rospy.is_shutdown():
            self.initial_angles_state = self.current_angles_state
            self.initial_strings_state = self.current_strings_state
            self.command = random.choice(possible_actions)
            self.send_motor_cmd()
            self.final_angles_state = self.current_angles_state
            self.final_strings_state = self.current_strings_state
            self.write_line(file)
        file.close()

    def joint_val_update(self, msg):
        self.current_angles_state = msg.data

    def tension_val_update(self, msg):
        for i in range(len(msg.data)):
            if msg.data[i] > max_tension:
                rospy.logfatal("you reach max tension")
                self.handle_tension_exception(i)
            if msg.data[i] < min_tension:
                rospy.logfatal("you reach min tension")
        self.current_strings_state = msg.data

    def send_motor_cmd(self):
        msg = Int32MultiArray(data=np.asarray(self.command).astype(int))
        self.motor_pwm_pub.publish(msg)
        time.sleep(1)
        self.stop_motor()

    def stop_motor(self):
        msg = Int32MultiArray(data= np.zeros(_N_string).astype(int))
        self.motor_pwm_pub.publish(msg)

    def handle_tension_exception(self, ind):
        self.command = np.zeros(_N_string)
        self.command[ind] = motor_vel_step_down
        msg = Int32MultiArray(data=self.command.astype(int))
        self.motor_pwm_pub.publish(msg)


def signal_handler(sig, frame):
    snake.stop_motor()
    print('You pressed Ctrl+C!')
    sys.exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rospy.init_node("snake", anonymous=True)
    rate = rospy.Rate(100)
    snake = SnakeExp()
    snake.run_experiment()
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C')
    signal.pause()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
