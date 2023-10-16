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
        self.INITIAL_PARTICLE_COUNT = 2000		# Number of particles to use
        self.POSITION_STD_DEV = 8
        self.ORIENTATION_STD_DEV = 8
        self.ODOM_ROTATION_NOISE = 0.01 		# Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.01 	# Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.01 			# Odometry y axis (side-side) noise
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict

        self.UPDATE_COORD_SD = 0.05           # Laser scan sampling noise
        self.UPDATE_ORIENT_SD = 0.2      # Laser scan orientation noise
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

        #initialising pose
        initial_orientation = initialpose.pose.pose.orientation
        initial_position = initialpose.pose.pose.position

        # ----- For each particle
        for i in range(self.INITIAL_PARTICLE_COUNT):
            # ----- Create a new pose
            new_pose = Pose()
            new_pose.position.x = initial_position.x + random.gauss(0, self.POSITION_STD_DEV)
            new_pose.position.y = initial_position.y + random.gauss(0, self.POSITION_STD_DEV)
            new_pose.orientation = rotateQuaternion(initial_orientation, random.gauss(0, self.ORIENTATION_STD_DEV))

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

        # ----- Do resampling
        new_particlecloud = PoseArray()
        new_particlecloud.poses = self.systematic_sampling(self.particlecloud.poses, normalised_weights, self.UPDATE_PARTICLE_COUNT)

        self.particlecloud = new_particlecloud

    def systematic_sampling(self, original_poses, weights, sample_count):
        sampled_poses = []
        random_range = random.uniform(0, 1.0 / sample_count)
        cumulative_weight = weights[0]
        weight_index = 0

        for sample_index in range(sample_count):
            sample_value = random_range + sample_index / sample_count

            while sample_value > cumulative_weight:
                weight_index += 1
                cumulative_weight += weights[weight_index]

            sampled_poses.append(self.add_noise(original_poses[weight_index]))

        return sampled_poses


    def sample_with_replacement(self, original_poses, weights, sample_count):
        sampled_poses = []

        for i in range(sample_count):
            # ----- Choose a random particle
            random_particle = np.random.choice(original_poses, p=weights)

            # ----- Add a copy of the random particle to the new particle cloud
            sampled_poses.append(self.add_noise(random_particle))

        return sampled_poses



    def add_noise(self, pose):
        """
        Add noise to a pose

        :Args:
            | pose (geometry_msgs.msg.Pose): pose to add noise to
        :Return:
            | (geometry_msgs.msg.Pose) new pose
        """
        coord_sd = self.UPDATE_COORD_SD
        orient_sd = self.UPDATE_ORIENT_SD

        # a 1 in 100 chance of having a big jump
        if random.randint(0, 100) == 0:
            coord_sd = 3
            orient_sd = 1

        new_pose = Pose()
        new_pose.position.x = pose.position.x + random.gauss(0, coord_sd)
        new_pose.position.y = pose.position.y + random.gauss(0, coord_sd)
        new_pose.orientation = rotateQuaternion(pose.orientation, random.gauss(0, orient_sd))
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
