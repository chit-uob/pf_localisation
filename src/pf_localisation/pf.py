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
        # self.ODOM_ROTATION_NOISE = 0.01 		# Odometry model rotation noise
        # self.ODOM_TRANSLATION_NOISE = 0.01 	# Odometry x axis (forward) noise
        # self.ODOM_DRIFT_NOISE = 0.01 			# Odometry y axis (side-side) noise
        self.KIDNAPPING_PROBABILITY = 0.005
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict

        self.UPDATE_COORD_SD = 0.1           # Laser scan sampling noise
        self.UPDATE_ORIENT_SD = 0.1      # Laser scan orientation noise
        self.UPDATE_PARTICLE_COUNT = 1000    # Number of particles to update

        self.weights = []


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
        print('IN INITIALISE')
        particlecloud = PoseArray()

        #initialising pose
        initial_orientation = initialpose.pose.pose.orientation
        initial_position = initialpose.pose.pose.position

        # ----- For each particle
        for i in range(self.INITIAL_PARTICLE_COUNT):
            # ----- Create a new pose
            print('initialising....')
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
        print('IN UPDATE')
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

        # Kidnapping: Introduce random particles
        num_kidnapped_particles = int(self.KIDNAPPING_PROBABILITY * len(self.particlecloud.poses))
        
        for _ in range(num_kidnapped_particles):
            kidnapped_pose = Pose()

            print('kidnappeddddddd')
            # Assuming some boundaries for the environment, set them accordingly
            kidnapped_pose.position.x = random.uniform(0, 10)
            kidnapped_pose.position.y = random.uniform(0, 10)
            random_yaw = random.uniform(-math.pi, math.pi)
            kidnapped_orientation = rotateQuaternion(Quaternion(), random_yaw)
            kidnapped_pose.orientation = kidnapped_orientation
            self.particlecloud.poses.append(kidnapped_pose)
        

        self.weights = weights
        self.particlecloud = new_particlecloud

    def systematic_sampling(self, original_poses, weights, sample_count):
        sampled_poses = []
        random_range = random.uniform(0, 1.0 / sample_count)
        cumulative_weight = weights[0]
        weight_index = 0

        for sample_index in range(sample_count):
            print('resampling...')
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
        if random.randint(0, 500) == 0:
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
        # Assume that self.particlecloud.poses is a list of Pose objects
        # and that self.weights is a list of corresponding weights

        # Select the top N% of particles based on weight
        N = 5  # for example, use the top 10% of particles
        num_particles = len(self.particlecloud.poses)
        num_top_particles = num_particles * N // 100
        print('IN ESTIMATE')
        # weights = []
        # # ----- For each particle
        # for pose in self.particlecloud.poses:
        #     weights.append(self.sensor_model.get_weight(scan, pose))
        
        # Get the indices of the top N% of particles based on weight
        top_particle_indices = sorted(range(num_particles), key=lambda i: self.weights[i], reverse=True)[:num_top_particles]
        
        # Compute the average position and orientation of the top particles
        avg_x = avg_y = avg_z = avg_w = 0.0
        for i in top_particle_indices:
            print('estimating....')
            avg_x += self.particlecloud.poses[i].position.x
            avg_y += self.particlecloud.poses[i].position.y
            q = self.particlecloud.poses[i].orientation
            avg_z += q.z
            avg_w += q.w
        
        avg_x /= num_top_particles
        avg_y /= num_top_particles
        avg_z /= num_top_particles
        avg_w /= num_top_particles
        
        # Normalize the average quaternion to ensure it's a unit quaternion
        norm = math.sqrt(avg_z**2 + avg_w**2)
        avg_z /= norm
        avg_w /= norm
        
        # Create a Pose object for the estimated pose
        estimated_pose = Pose()
        estimated_pose.position.x = avg_x
        estimated_pose.position.y = avg_y
        estimated_pose.orientation.z = avg_z
        estimated_pose.orientation.w = avg_w
        
        return estimated_pose
