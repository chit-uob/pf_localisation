from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random

from time import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.01 		# Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.01 	# Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.01 			# Odometry y axis (side-side) noise
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        self.INITIAL_NOISE = 5                # Noise in initial particle cloud
        self.SCAN_SAMPLE_NOISE = 1           # Laser scan sampling noise
        
       
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
        self.particlecloud = PoseArray()
        self.particlecloud.header.frame_id = 'map'
        self.particlecloud.header.stamp = rospy.Time.now()
        self.particlecloud.poses = []

        #initialising pose
        self.initialpose = initialpose

        # ----- For each particle
        for i in range(100):
            # ----- Create a new pose
            new_pose = Pose()
            new_pose.position.x = initialpose.pose.pose.position.x + (random() * 1 - 0.5) * self.INITIAL_NOISE
            new_pose.position.y = initialpose.pose.pose.position.y + (random() * 1 - 0.5) * self.INITIAL_NOISE
            new_pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, (random() * 1 - 0.5) * self.INITIAL_NOISE)
            self.particlecloud.poses.append(new_pose)

        return self.particlecloud
 
    
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
        weights = [w / weight_sum for w in weights]

        # ----- Resample
        new_particles = []

        # ----- Pick 100 particles from the old ones, with replacement
        for i in range(100):
            # ----- Pick a random particle, weighted by the weights
            r = random()
            total = 0.0
            for j in range(len(weights)):
                total += weights[j]
                if total > r:
                    # ----- add noise to the particle
                    new_pose = Pose()
                    new_pose.position.x = self.particlecloud.poses[j].position.x + (random() * 0.1 - 0.05) * self.SCAN_SAMPLE_NOISE
                    new_pose.position.y = self.particlecloud.poses[j].position.y + (random() * 0.1 - 0.05) * self.SCAN_SAMPLE_NOISE
                    new_pose.orientation = rotateQuaternion(self.particlecloud.poses[j].orientation, (random() * 0.1 - 0.05) * self.SCAN_SAMPLE_NOISE)
                    new_particles.append(new_pose)
                    break

        self.particlecloud.poses = new_particles

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
