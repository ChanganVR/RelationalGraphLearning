#!/usr/bin/env python
# license removed for brevity
import rospy
import math
import numpy as np

import tf2_ros

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from people_msgs.msg import People

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
    self.vel_angular = History()
    self.vel_linear = History()
    self.goal_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.new_goal_callback)
    self.people_sub = rospy.Subscriber("people", People, self.people_callback)
    self.odom_sub = rospy.Subscriber("pose", Odometry, self.odom_callback)
    self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)   
		self.laser_frame = "laser"
		self.odom_frame = "odom"
		self.tfBuffer = tf2_ros.Buffer()
    self.listener = tf2_ros.TransformListener(self.tfBuffer)

		
  def people_callback(self, people_msg):
    people_list = []
    # self.goal (x,y)
    # robot_vel (velx ,y)
    for person in people_msg.people:
      transform = self.tfBuffer.lookup_transform(self.laser_frame, people_msgs.header.frame_id, rospy.Time(0), rospy.Duration(0.5))
      pose_transformed = PointStamped()
      pose_transformed.point = goal.pos
      pose_transformed.header  = goal.header
      pose_transformed = tf2_geometry_msgs.do_transform_point(pose_transformed, transform)
      new_destination = PoseStamped() 
      people_list.append((person.position.x, person.position.y, person.velocity.x, person.velocity.y))
     
    # rospy.loginfo(people_list)
    # speed, heading = self.net(people_list, self.goal, robot_vel)
  
  def odom_callback(self, odom_msg):
    self.vel_angular.add(odom_msg.twist.twist.angular.z)
    self.vel_linear.add(odom_msg.twist.twist.linear.x)
    vel_angular = self.vel_angular.get_avg()
    vel_linear = self.vel_linear.get_avg()
    self.robot_vel = (math.sin(vel_angular)*vel_linear, math.cos(vel_angular)*vel_linear)
    rospy.loginfo("vel is {:1.4f}, {:1.4f}".format(self.robot_vel[0], self.robot_vel[1]))

  def new_goal_callback(self, goal_msg):
    rospy.loginfo("got new goal pose: {}".format(goal_msg.pose))
    self.goal = goal_msg.pose

  def send_acction(self, speed, heading):
    rospy.loginfo("got command speed: {} heading: {}".format(speed, heading))
    cmd_vel = Twist()
    cmd_vel.linear.x = min(speed/8.0, 0.5)
    if heading < math.pi:
      angular_speed = -heading
    else:
      angular_speed = 2 * math.pi - heading

    cmd_vel.angular.z = min(angular_speed, 1.75)
    self.cmd_vel_pub.publish(cmd_vel)

if __name__ == '__main__':
  rospy.init_node("crowd_nav")
  crowd_nav = CrowdNav()
  rospy.spin()
