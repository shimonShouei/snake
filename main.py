import os
import random
import subprocess
import time
import signal
import sys
import shlex
from psutil import Popen
import numpy as np
import itertools
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray

_max_PWM = 60
_max_angle = 30
_N_joints = 4
_N_string = _N_joints * 2
max_tension = 10
min_tension = 0.3
max_angl = 20
min_angl = -20

motor_vel_step_up = 100
motor_vel_step_down = -100
delay = 0.2
repeat = 8
possible_actions = list(map(list, itertools.product([motor_vel_step_up, motor_vel_step_down], repeat=repeat)))


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
        self.file = open('output.txt', 'a')
        self.file.write("initial_angles_state$initial_strings_state$command$final_angles_state$final_strings_state\n")


    def write_line(self, file):
        file.write(
            self.initial_angles_state.__str__() + '$' + self.initial_strings_state.__str__() + '$' + self.command.data.__str__() + '$' + self.final_angles_state.__str__() + '$' + self.final_strings_state.__str__())
        file.write("\n")

    def run_experiment(self):
        while not rospy.is_shutdown():
            for i in range(100):
                # self.initial_angles_state = self.current_angles_state
                # self.initial_strings_state = self.current_strings_state
                self.middlecheck()
                self.command = random.choice(possible_actions)
                self.command = Int32MultiArray(data=np.asarray(self.command).astype(int))
                self.send_motor_cmd()
                # self.final_angles_state = self.current_angles_state
                # self.final_strings_state = self.current_strings_state
                # self.write_line(file)
                time.sleep(0.2)
            self.stop_motor()
        self.file.close()

    def joint_val_update(self, msg):
        # print("msg_data: ", msg.data)
        self.current_angles_state = msg.data
        # print("current_joint: ", self.current_angles_state)

    def tension_val_update(self, msg):
        # print(msg)
        self.current_strings_state = msg.data

    def send_motor_cmd(self, d=delay):
        self.initial_angles_state = self.current_angles_state
        self.initial_strings_state = self.current_strings_state
        self.motor_pwm_pub.publish(self.command)
        time.sleep(d)
        self.pub_zero()
        self.final_angles_state = self.current_angles_state
        self.final_strings_state = self.current_strings_state
        self.write_line(self.file)

    def stop_motor(self):
        node_process = Popen(
            shlex.split('rosrun robot_snake_4 controller_v4_3'), shell=False
        )
        time.sleep(8)

        # stop
        node_process.terminate()

    def pub_zero(self):
        msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
        self.motor_pwm_pub.publish(msg)

    def add_tension(self, ind):
        while self.current_strings_state[ind] < min_tension:
            msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
            msg.data[ind] = motor_vel_step_up
            self.command = msg
            self.send_motor_cmd()

    def reduce_tension(self, ind):
        while self.current_strings_state[ind] > max_tension:
            msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
            msg.data[ind] = motor_vel_step_down
            self.command = msg
            self.send_motor_cmd()

    def add_angle(self, ind):
        msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
        msg.data[ind] = motor_vel_step_up
        msg.data[ind+1] = motor_vel_step_down
        self.command = msg
        self.send_motor_cmd()

    def reduce_angle(self, ind):
        msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
        msg.data[ind] = motor_vel_step_down
        msg.data[ind+1] = motor_vel_step_up
        self.command = msg
        self.send_motor_cmd(0.1)

    def middlecheck(self):
        for i in range(len(self.current_strings_state)-1):
            if self.current_strings_state[i] > max_tension:
                rospy.logfatal("you reach max tension")
                self.reduce_tension(i)

            if self.current_strings_state[i] < min_tension:
                rospy.logfatal("you reach min tension")
                self.add_tension(i)

        for i in range(len(self.current_angles_state)-1):
            if self.current_angles_state[i] > max_angl:
                rospy.logfatal("you reach max angle")
                # self.reduce_tension(2*i)
                # self.add_tension(2*i+1)
                self.reduce_angle(i)

            if self.current_angles_state[i] < min_angl:
                rospy.logfatal("you reach min angle")
                # self.add_tension(2*i)
                # self.reduce_tension(2*i+1)
                self.add_angle(i)

# def signal_handler(sig, frame):
#     snake.stop_motor()
#     print('You pressed Ctrl+C!')
#     sys.exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rospy.init_node("snake", anonymous=True)
    rate = rospy.Rate(100)
    snake = SnakeExp()
    snake.run_experiment()
    # signal.signal(signal.SIGINT, signal_handler)
    # print('Press Ctrl+C')
    # signal.pause()