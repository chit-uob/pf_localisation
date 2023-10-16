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
 
        # # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        # self.INITIAL_NOISE = 5                # Noise in initial particle cloud
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
        particlecloud = PoseArray()
        # self.particlecloud.header.frame_id = 'map'
        # self.particlecloud.header.stamp = rospy.Time.now()
        # self.particlecloud.poses = []

        #initialising pose
        initial_orientation = initialpose.pose.pose.orientation
        initial_position = initialpose.pose.pose.position

        # ----- For each particle
        for i in range(2000):
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
        weights = [w / weight_sum for w in weights]

        # ----- Resample
        new_particles = []

        # ----- Pick 100 particles from the old ones, with replacement
        for i in range(1000):
            # ----- Pick a random particle, weighted by the weights
            r = random.random()
            total = 0.0
            for j in range(len(weights)):
                total += weights[j]
                if total > r:
                    # ----- add noise to the particle
                    new_pose = Pose()
                    new_pose.position.x = self.particlecloud.poses[j].position.x + (random.random() * 0.1 - 0.05) * self.SCAN_SAMPLE_NOISE
                    new_pose.position.y = self.particlecloud.poses[j].position.y + (random.random() * 0.1 - 0.05) * self.SCAN_SAMPLE_NOISE
                    new_pose.orientation = rotateQuaternion(self.particlecloud.poses[j].orientation, (random.random() * 0.1 - 0.05) * self.SCAN_SAMPLE_NOISE)
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

        # Assume that self.particlecloud.poses is a list of Pose objects
        # and that self.weights is a list of corresponding weights

        # weights = []
        # for pose in self.particlecloud.poses:
        #     weights.append(self.sensor_model.get_weight(scan, pose))

        # Select the top N% of particles based on weight
        # N = 10  # for example, use the top 10% of particles
        # num_particles = len(self.particlecloud.poses)
        # num_top_particles = num_particles * N // 100
        #
        # # Get the indices of the top N% of particles based on weight
        # top_particle_indices = sorted(range(num_particles), key=lambda i: weights[i], reverse=True)[
        #                        :num_top_particles]
        #
        # # Compute the average position and orientation of the top particles
        # avg_x = avg_y = avg_z = avg_w = 0.0
        # for i in top_particle_indices:
        #     avg_x += self.particlecloud.poses[i].position.x
        #     avg_y += self.particlecloud.poses[i].position.y
        #     q = self.particlecloud.poses[i].orientation
        #     avg_z += q.z
        #     avg_w += q.w
        #
        # avg_x /= num_top_particles
        # avg_y /= num_top_particles
        # avg_z /= num_top_particles
        # avg_w /= num_top_particles
        #
        # # Normalize the average quaternion to ensure it's a unit quaternion
        # norm = math.sqrt(avg_z ** 2 + avg_w ** 2)
        # avg_z /= norm
        # avg_w /= norm
        #
        # # Create a Pose object for the estimated pose
        # estimated_pose = Pose()
        # estimated_pose.position.x = avg_x
        # estimated_pose.position.y = avg_y
        # estimated_pose.orientation.z = avg_z
        # estimated_pose.orientation.w = avg_w
        #
        # return estimated_pose

        #Select the top N% of particles based on weight
        # N = 10  # for example, use the top 10% of particles
        # num_particles = len(self.particlecloud.poses)
        # num_top_particles = num_particles * N // 100

        for i in self.particlecloud.poses:
            average_x += self.particlecloud.poses[i].position.x
            average_y += self.particlecloud.poses[i].position.y
            q = self.particlecloud.poses[i].orientation
            average_z += q.z
            average_w += q.w

        average_x /= len(self.particlecloud.poses)
        average_y /= len(self.particlecloud.poses)
        average_z /= len(self.particlecloud.poses)
        average_w /= len(self.particlecloud.poses)

        # Normalize the average quaternion to ensure it's a unit quaternion
        norm = math.sqrt(avg_z ** 2 + avg_w ** 2)
        avg_z /= norm
        avg_w /= norm

        # Create a Pose object for the estimated pose
        estimated_pose = Pose()
        estimated_pose.position.x = avg_x
        estimated_pose.position.y = avg_y
        estimated_pose.orientation.z = avg_z
        estimated_pose.orientation.w = avg_w

        return estimated_pose

        def estimate_pose(self):

        """
        Abbas
        """
        num_particles = len(self.particlecloud.poses)

        if num_particles == 0:
            # No particles available, return an invalid pose
            estimated_pose = Pose()
            estimated_pose.position.x = float('nan')
            estimated_pose.position.y = float('nan')
            estimated_pose.orientation = Quaternion()
            estimated_pose.orientation.w = float('nan')
            return estimated_pose

        # Initialize variables to accumulate the sum of positions and orientations
        sum_x, sum_y, sum_quaternion = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)

        # Sum the positions and orientations of all particles
        for pose in self.particlecloud.poses:
            sum_x += pose.position.x
            sum_y += pose.position.y
            sum_quaternion = (
                sum_quaternion[0] + pose.orientation.x,
                sum_quaternion[1] + pose.orientation.y,
                sum_quaternion[2] + pose.orientation.z,
                sum_quaternion[3] + pose.orientation.w
            )

        # Calculate the average position and orientation
        average_x = sum_x / num_particles
        average_y = sum_y / num_particles
        average_quaternion = (
            sum_quaternion[0] / num_particles,
            sum_quaternion[1] / num_particles,
            sum_quaternion[2] / num_particles,
            sum_quaternion[3] / num_particles
        )

        # Create the estimated pose with the average values
        estimated_pose = Pose()
        estimated_pose.position.x = average_x
        estimated_pose.position.y = average_y
        estimated_pose.orientation = Quaternion(*average_quaternion)

        return estimated_pose




        # for pose in self.particlecloud.poses:
        #     average_x = sum(pose.position.x) / len(self.particlecloud.poses)
        #     average_y = sum(pose.position.y) / len(self.particlecloud.poses)
        #     q = self.particlecloud.poses[].orientation

        #     average_heading_x = sum(math.cos(getHeading(pose.position.x))) / len(self.particlecloud.poses)
        #     average_heading_y = sum(math.sin(getHeading(pose.position.x))) / len(self.particlecloud.poses)
        #     average_heading = math.atan2(average_heading_y, average_heading_x)

        # # Create a new Pose message for the estimated pose
        # estimated_pose = Pose()
        # estimated_pose.position.x = average_x
        # estimated_pose.position.y = average_y
        # estimated_pose.orientation = rotateQuaternion(Quaternion(), average_heading)

        # return estimated_pose

        #return self.particlecloud.poses[0]
