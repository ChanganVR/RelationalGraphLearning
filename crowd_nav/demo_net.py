#!/usr/bin/env python
# license removed for brevity
import rospy
import math
import numpy as np
import os

import tf2_ros
#import tf2_geometry_msgs

from std_msgs.msg import String
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from people_msgs.msg import People

import logging
import argparse
import importlib.util
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.state import JointState

def quaternion_to_euler(x, y, z, w):
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(t0, t1)
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch = math.asin(t2)
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(t3, t4)
  return [yaw, pitch, roll]


class History:
  def __init__(self, window_size=3):
    self.window_size = window_size
    self.list = [0 for x in range(window_size)]
    self.idx = 0

  def add(self, item):
    self.list[self.idx] = item
    self.idx = (self.idx + 1) % self.window_size

  def get_avg(self):
    return (np.mean(self.list))

class CrowdNav:
  def __init__(self):
    self.goal = None
    self.robot_pos = (0,0,0)
    self.is_safe = False
    self.robot_vel = (0,0)
    self.vel_angular = History()
    self.last_reset = rospy.Time.now().to_sec() - 4
    self.vel_linear = History()
    self.goal_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.new_goal_callback)
    self.joy_sub = rospy.Subscriber("joy", Joy, self.joy_callback)
    self.people_sub = rospy.Subscriber("people", People, self.people_callback)
    self.odom_sub = rospy.Subscriber("pose", Odometry, self.odom_callback)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)   
    self.laser_frame = "laser"
    self.odom_frame = "odom"
    self.tfBuffer = tf2_ros.Buffer()
    self.listener = tf2_ros.TransformListener(self.tfBuffer)
    self.init_net()
    self.cmd_vel = Twist()

  def get_goal(self):
    if self.goal is None:
      return None
    return (self.goal[0], self.goal[1])
    # transform = self.tfBuffer.lookup_transform(self.laser_frame, self.goal.header.frame_id, rospy.Time(0), rospy.Duration(0.5))
    # pose_transformed = PointStamped()
    # pose_transformed.point = self.goal.pos
    # pose_transformed.header  = self.goal.header
    # pose_transformed = self.listener.transformPoint(self.laser_frame, pose_transformed)
    # return (pose_transformed.point.x, pose_transformed.point.y) 
    
  def people_callback(self, people_msg):
    people_list = []
    # self.goal (x,y)
    # robot_vel (velx ,y)
    for person in people_msg.people:
      # transform = self.tfBuffer.lookup_transform(self.laser_frame, people_msg.header.frame_id, rospy.Time(0), rospy.Duration(0.5))
      # pose_transformed = PointStamped()
      # pose_transformed.point = person.position
      # pose_transformed.header  = people_msg.header
      # pose_transformed = tf2_geometry_msgs.do_transform_point(pose_transformed, transform)
      # pose_transformed = self.listener.transformPoint(self.laser_frame, pose_transformed)
      people_list.append((person.position.x, person.position.y, 10 * person.velocity.x, 10 * person.velocity.y))
    #if len(people_list) == 0:
    # while len(people_list) > 5:
    #   min_dist = 1000
    #   for person in people_list:
    #     math.hypot
    #   people_list = people_list[:5]
    # people_list = [(10,10,0.6,0.7),(7,3,0.6,0.8),(7,10,0.6,0.8),(-15,-5,-0.6,-0.8),(-5,-5,-0.6,-0.8) ]
    if len (people_list) == 0:
      people_list = [(10,10,0.6,0.7)]
    goal = self.get_goal()
    if goal is None or self.robot_vel is None:
      rospy.logwarn("goal: {} or robot_vel: {} is None".format(goal, self.robot_vel))
      return
    action = self.demo_net(people_list, goal, self.robot_vel)
    conver_angle_global =  math.atan2(action[1], action[0]) #- math.pi / 2
    angular_vel = (conver_angle_global - self.robot_pos[2] + 2 * math.pi) % (2*math.pi)
    linear_vel = max(abs(action[0]), abs(action[1]))
    rospy.loginfo("action: {}, goal: {}, pos: {}, gdeg: {}".format(action, self.get_goal(), self.robot_pos, conver_angle_global))
    rospy.loginfo("angular_vel: {}, linear_vel: {} action: {}".format(angular_vel*180/math.pi, linear_vel, action))
    if math.hypot(goal[0] - self.robot_pos[0], goal[1] - self.robot_pos[1]) < 1:
      rospy.logwarn("GOAL REACHED")
      self.cmd_vel_pub.publish(Twist())
      return
    self.send_action(linear_vel, angular_vel) 

  def reinit_p2os(self):
    if rospy.Time.now().to_sec() -  self.last_reset < 10:
      return
    self.last_reset = rospy.Time.now().to_sec()
    os.popen("rosnode kill p2os")
    time.sleep(2)
    os.popen('rosrun p2os_driver p2os_driver _port:="/dev/ttyUSB5"')
    #os.popen('rosrun p2os_driver p2os_driver _port:="/dev/ttyUSB1"')
    os.popen('rostopic pub /cmd_motor_state p2os_msgs/MotorState "state: 1" --once')
  
  def joy_callback(self, joy_msg):
    if joy_msg.buttons[2] == 1:
      self.reinit_p2os()
    elif joy_msg.buttons[3] == 1:
      self.goal = (8, 0) 
    if joy_msg.buttons[1] == 1:
      self.is_safe = True
      rospy.logwarn("safe")
    else:
      self.is_safe = False
      rospy.logwarn("not safe")

  def odom_callback(self, odom_msg):
    self.vel_angular.add(odom_msg.twist.twist.angular.z)
    self.vel_linear.add(odom_msg.twist.twist.linear.x)
    vel_angular = self.vel_angular.get_avg()
    vel_linear = self.vel_linear.get_avg()
    self.robot_vel = (math.sin(vel_angular)*vel_linear, math.cos(vel_angular)*vel_linear)
    euler_angle = quaternion_to_euler(odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w)
    self.robot_pos = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, euler_angle[0])
    # rospy.loginfo("vel is {:1.4f}, {:1.4f} robot pos: {}".format(self.robot_vel[0], self.robot_vel[1], self.robot_pos))

  def new_goal_callback(self, goal_msg):
    # rospy.loginfo("got new goal pose: {}".format(goal_msg.pose))
    self.goal = (goal_msg.pose.position.x, goal_msg.pose.position.y)

  def send_action(self, speed, heading):
    # rospy.loginfo("got command speed: {} heading: {}".format(speed, heading))
    cmd_vel = Twist()
    if heading < math.pi:
      angular_speed = heading
    else:
      angular_speed = - 2 * math.pi + heading
  
    if abs (angular_speed) < 0.6:
      cmd_vel.linear.x =  min(abs(speed/2.0), 0.3)

    if abs(angular_speed) < 0.01:
      angular_speed = 0
      cmd_vel.linear.x = 0.5 # min(abs(speed/8.0), 0.5)
    
    cmd_vel.angular.z = min(angular_speed, 1.75)
    rospy.loginfo("angular: {} linear {}".format(angular_speed, cmd_vel.linear.x))
    if self.is_safe:
      self.cmd_vel_pub.publish(cmd_vel)
    else:
      rospy.logerr("not safe")
      self.cmd_vel_pub.publish(Twist())
  
  def init_net(self):
    rospy.loginfo("start init net")
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='model_predictive_rl')
    parser.add_argument('-m', '--model_dir', type=str, default='/home/autolab/Downloads/mp_separate_l2_d2_w4')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--group_size', type=int, default=None)
    parser.add_argument('--group_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=1)
    parser.add_argument('-w', '--planning_width', type=int, default=1)
    parser.add_argument('--sparse_search', default=False, action='store_true')
  
    args = parser.parse_args()
  
    # configure logging and device
 
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    model_dir = '/local-scratch/changan/icra_benchmark_15000/mp_separate_l2_d2_w4'
    if args.model_dir is not None:
      config_file = os.path.join(args.model_dir, 'config.py')
      model_weights = os.path.join(args.model_dir, 'best_val.pth')
      logging.info('Loaded RL weights with best VAL')
  
    # configure policy
    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    policy_config = config.PolicyConfig(args.debug)
    self.policy = policy_factory[policy_config.name]()
    if args.planning_depth is not None:
      policy_config.model_predictive_rl.do_action_clip = True
      policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
      policy_config.model_predictive_rl.do_action_clip = True
      policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
      policy_config.model_predictive_rl.sparse_search = True
  
    self.policy.configure(policy_config)
    if self.policy.trainable:
      if args.model_dir is None:
        parser.error('Trainable policy must be specified with a model weights directory')
      self.policy.load_model(model_weights)
  
    # configure environment
    env_config = config.EnvConfig(args.debug)
  
    if args.human_num is not None:
      env_config.sim.human_num = args.human_num
    if args.group_num is not None:
      env_config.sim.group_num = args.group_num
    if args.group_size is not None:
      env_config.sim.group_size = args.group_size
  
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    self.robot = Robot(env_config, 'robot')
    self.robot.time_step = env.time_step
    self.robot.set_policy(self.policy)
  
    self.train_config = config.TrainConfig(args.debug)
    self.epsilon_end = self.train_config.train.epsilon_end
    if not isinstance(self.robot.policy, ORCA):
      self.robot.policy.set_epsilon(self.epsilon_end)
  
    self.policy.set_phase('test')
    self.policy.set_device(device)
    self.policy.set_env(env)
    self.robot.print_info()
    humans = [(11, 2, 1, 3)]
    goal = (1, 2)
    robot_vel = (3, 6)
    start = time.time()
    # rospy.loginfo("before call")
    vx, vy = self.demo_net(humans, goal, robot_vel)

    rospy.loginfo("vs,vy: {} {} time: {}".format(vx, vy, time.time() - start))


  def demo_net(self, humans, goal, robot_vel):
  
    '''
    do some processing for the raw observation from the robot
    '''
    ob = []
    for human in humans:
      # human observable state: self.px, self.py, self.vx, self.vy, self.radius
      # robot full state: (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)
      ob.append(ObservableState(-human[1], human[0], -human[3], human[2], 0.3))
  
    """
    set the robot state from sensor data
    """
    self.robot.px = -self.robot_pos[1]
    self.robot.py = self.robot_pos[0]
    self.robot.vx = -robot_vel[1]
    self.robot.vy = robot_vel[0]
    # self.robot.vx = math.sin(self.robot_pos[2])
    # self.robot.vy = math.sin(self.robot_pos[2])
    self.robot.theta = 1.5707963267948966
    #self.robot.theta = self.robot_pos[2]
    self.robot.gx = -goal[1]
    self.robot.gy = goal[0]
    action = self.robot.act(ob)
    rospy.loginfo("robot pos: {} {}, goal {} {} action: {}".format(self.robot.px, self.robot.py, self.robot.gx, self.robot.gy, action))
    return action.vy, -action.vx


if __name__ == '__main__':
  rospy.init_node("crowd_nav")
  crowd_nav = CrowdNav()
  rospy.spin()

