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

 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
       
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
        partical_cloud = PoseArray()
        partical_cloud.poses = [initialpose for _ in range(self.NUMBER_PREDICTED_READINGS)]
        

        #Adding noise
        self.particlecloud = partical_cloud
        return partical_cloud

 
    
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

        # ----- For each particle
        for i in range(100):
            # ----- Choose a random particle
            r = random()
            c = 0
            for j in range(len(weights)):
                c += weights[j]
                if c > r:
                    selected_pose = self.particlecloud.poses[j]

                    # ----- Add noise
                    selected_pose.position.x += random() * 0.1
                    selected_pose.position.y += random() * 0.1
                    selected_pose.orientation = rotateQuaternion(selected_pose.orientation, random() * 0.1)

                    new_particles.append(selected_pose)
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
        pass
