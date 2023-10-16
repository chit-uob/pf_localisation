from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
import random

from time import time
import numpy as np


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.POSITION_STD_DEV = 8
        self.ORIENTATION_STD_DEV = 8
        # self.ODOM_ROTATION_NOISE = 0.01 		# Odometry model rotation noise
        # self.ODOM_TRANSLATION_NOISE = 0.01 	# Odometry x axis (forward) noise
        # self.ODOM_DRIFT_NOISE = 0.01 			# Odometry y axis (side-side) noise
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        # self.INITIAL_NOISE = 5                # Noise in initial particle cloud
        self.SCAN_SAMPLE_NOISE = 1           # Laser scan sampling noise
        self.INITIAL_NOISE = 5                # Noise in initial particle cloud
        self.SCAN_SAMPLE_NOISE = 0.1           # Laser scan sampling noise
        self.UPDATE_PARTICLE_COUNT = 1000    # Number of particles to update


    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        #initialising array
        particlecloud = PoseArray()
        # self.particlecloud.header.frame_id = 'map'
        # self.particlecloud.header.stamp = rospy.Time.now()
        # self.particlecloud.poses = []

        #initialising pose
        initial_orientation = initialpose.pose.pose.orientation
        initial_position = initialpose.pose.pose.position

        # ----- For each particle
        for i in range(1000):
            # ----- Create a new pose
            new_pose = Pose()
            new_pose.position.x = initial_position.x + random.gauss(0, self.POSITION_STD_DEV)
            new_pose.position.y = initial_position.y + random.gauss(0, self.POSITION_STD_DEV)
            # new_pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, (random() * 1 - 0.5) * self.INITIAL_NOISE)

            initial_yaw = getHeading(initial_orientation)
            noisy_yaw = initial_yaw + random.gauss(0, self.ORIENTATION_STD_DEV)

            q_noisy = rotateQuaternion(initial_orientation, noisy_yaw)
            new_pose.orientation = q_noisy

            particlecloud.poses.append(new_pose)

        return particlecloud

    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        weights = []
        # ----- For each particle
        for pose in self.particlecloud.poses:
            weights.append(self.sensor_model.get_weight(scan, pose))

        # ----- Normalise weights
        weight_sum = sum(weights)
        normalised_weights = [weight / weight_sum for weight in weights]

        # ----- Randomly sample particles from the particle cloud with replacement
        new_particlecloud = PoseArray()

        for i in range(self.UPDATE_PARTICLE_COUNT):
            # ----- Choose a random particle
            random_index = np.random.choice(len(self.particlecloud.poses), p=normalised_weights)
            random_particle = self.particlecloud.poses[random_index]

            # ----- Add a copy of the random particle to the new particle cloud
            new_particlecloud.poses.append(self.add_noise(random_particle))

        self.particlecloud = new_particlecloud




    def add_noise(self, pose):
        """
        Add noise to a given pose.

        :Args:
            | pose (geometry_msgs.msg.Pose): pose to add noise to
        :Return:
            | (geometry_msgs.msg.Pose) pose with noise added
        """
        new_pose = Pose()
        new_pose.position.x = pose.position.x + (random.random() * 1 - 0.5) * self.SCAN_SAMPLE_NOISE
        new_pose.position.y = pose.position.y + (random.random() * 1 - 0.5) * self.SCAN_SAMPLE_NOISE
        new_pose.orientation = rotateQuaternion(pose.orientation, (random.random() * 1 - 0.5) * self.SCAN_SAMPLE_NOISE)
        return new_pose


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
        return self.particlecloud.poses[0]
