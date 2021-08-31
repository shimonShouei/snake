import logging
import os
import random
import subprocess
import time
import signal
import sys
import shlex
from datetime import date
import pandas as pd
import tqdm as tqdm
from psutil import Popen
import numpy as np
import itertools
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import roslaunch
import shlex
import signal
import time


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


_max_PWM = 60
_max_angle = 30
_N_joints = 4
_N_string = _N_joints * 2
max_tension = 10
min_tension = 0.3
max_angl = 15
min_angl = -15

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
        self.current_angles_state = (-1, -1, -1, -1)
        self.current_strings_state = (-1, -1, -1, -1, -1, -1, -1, -1)
        self.initial_angles_state = (-1, -1, -1, -1)
        self.initial_strings_state = (-1, -1, -1, -1, -1, -1, -1, -1)
        self.final_angles_state = (-1, -1, -1, -1)
        self.final_strings_state = (-1, -1, -1, -1, -1, -1, -1, -1)
        self.command = []
        self.file_name = "./Data/data" + date.today().__str__() + ".csv"
        self.data = pd.DataFrame(
            columns=["initial_j_1", "initial_j_2", "initial_j_3", "initial_j_4", "initial_s_1", "initial_s_2",
                     "initial_s_3", "initial_s_4", "initial_s_5", "initial_s_6", "initial_s_7", "initial_s_8",
                     "command_1", "command_2", "command_3", "command_4", "command_5", "command_6", "command_7",
                     "command_8", "final_j_1", "final_j_2", "final_j_3", "final_j_4", "final_s_1", "final_s_2",
                     "final_s_3", "final_s_4", "final_s_5", "final_s_6", "final_s_7", "final_s_8"])
        self.data.to_csv(self.file_name)

    def run_experiment(self):
        while not rospy.is_shutdown() and not killer.kill_now:
            for i in tqdm.tqdm(range(100)):
                self.middle_check()
                self.command = random.choice(possible_actions)
                self.command = Int32MultiArray(data=np.asarray(self.command).astype(int))
                self.send_motor_cmd()
                time.sleep(0.2)

            self.stop_motor()

            self.data.to_csv(self.file_name, mode='a', header=False)
            self.data = self.data[0:0]
        self.pub_zero()

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
        temp_data = self.initial_angles_state + self.initial_strings_state + tuple(
            self.command.data) + self.final_angles_state + \
                    self.final_strings_state
        a_series = pd.Series(temp_data, index=self.data.columns)
        self.data = self.data.append(a_series, ignore_index=True)

    def stop_motor(self):
        # node_process = Popen(
        #     shlex.split('#!/bin/bash rosrun robot_snake_4 controller_v4_3 ')
        # )
        # subprocess.Popen(
        #     '#!/bin/bash rosrun robot_snake_4 controller_v4_3'.split(' '), shell=True
        # )
        proc = subprocess.call("~/PycharmProjects/snake/controller.sh", shell=True)

        # node =roslaunch.core.Node("robot_snake_4", "controller_v4_3")
        # launch = roslaunch.scriptapi.ROSLaunch()
        # launch.start()
        #
        # process = launch.launch(node)
        # print(process.is_alive())
        # time.sleep(8)
        # process.stop()

        # stop
        # node_process.terminate()

    def pub_zero(self):
        msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
        self.motor_pwm_pub.publish(msg)
        rospy.loginfo("Published zero")

    def add_tension(self, ind):
        while self.current_strings_state[ind] < min_tension:
            msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
            msg.data[ind] = motor_vel_step_up
            self.command = msg
            self.send_motor_cmd(0.3)
            rospy.loginfo("Adding tension")

    def reduce_tension(self, ind):
        while self.current_strings_state[ind] > 6:
            msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
            msg.data[ind] = motor_vel_step_down
            self.command = msg
            self.send_motor_cmd(0.3)
            rospy.loginfo("Reducing tension")

    def add_angle(self, ind):
        msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
        msg.data[ind] = motor_vel_step_up
        msg.data[ind + 1] = motor_vel_step_down
        self.command = msg
        self.send_motor_cmd(0.3)
        rospy.loginfo("Adding angle")

    def reduce_angle(self, ind):
        msg = Int32MultiArray(data=np.zeros(_N_string).astype(int))
        msg.data[ind] = motor_vel_step_down
        msg.data[ind + 1] = motor_vel_step_up
        while self.current_angles_state[ind] > 6:
            self.command = msg
            self.send_motor_cmd(0.3)
            rospy.loginfo("Reducing angle")
            rospy.loginfo(self.current_angles_state[ind])

    def middle_check(self):
        for i in range(len(self.current_strings_state) - 1):
            if self.current_strings_state[i] > max_tension:
                rospy.logwarn("you reach max tension")
                self.reduce_tension(i)

            if self.current_strings_state[i] < min_tension:
                rospy.logwarn("you reach min tension")
                self.add_tension(i)

        for i in range(len(self.current_angles_state) - 1):
            if self.current_angles_state[i] > max_angl:
                rospy.logwarn("you reach max angle")
                # self.reduce_tension(2*i)
                # self.add_tension(2*i+1)
                self.reduce_angle(i)

            if self.current_angles_state[i] < min_angl:
                rospy.logwarn("you reach min angle")
                # self.add_tension(2*i)
                # self.reduce_tension(2*i+1)
                self.add_angle(i)


# def signal_handler(sig, frame):
#     snake.stop_motor()
#     print('You pressed Ctrl+C!')
#     sys.exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s')
    rospy.init_node("snake", anonymous=True)
    killer = GracefulKiller()
    rate = rospy.Rate(100)
    snake = SnakeExp()
    snake.run_experiment()
    # signal.signal(signal.SIGINT, signal_handler)
    # print('Press Ctrl+C')
    # signal.pause()
