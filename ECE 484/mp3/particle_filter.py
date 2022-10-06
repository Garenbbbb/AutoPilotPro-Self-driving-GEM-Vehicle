import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode
import bisect

# python main.py --num_particles 1000 --sensor_limit 15
def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()
        for i in range(num_particles):
            x = np.random.uniform(0, world.width)
            y = np.random.uniform(0, world.height)
            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))
        self.particles = np.array(particles)          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        self.last_integration = 0

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self, x1, x2, std = 5000):
        tmp1 = np.array(x1)
        tmp2 = np.array(x2)
        return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))

    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####

        for particle in self.particles:
            reading_particle = particle.read_sensor()
            w = self.weight_gaussian_kernel(reading_particle, readings_robot)
            particle.weight = w

        s = sum([p.weight for p in self.particles]) 																																																																																											          
        print("weight sum before:", s)

        for particle in self.particles:
            particle.weight = particle.weight / s

        print("weight sum after", sum([p.weight for p in self.particles]))


    def makeParticle(self, idx):
        oldParticle = self.particles[idx]
        
        if np.random.binomial(n=1, p=0, size=1):
            x = np.random.uniform(0, self.world.width)
            y = np.random.uniform(0, self.world.height)
            w = np.random.uniform(0, 1)
            newParticle = Particle(x=x, y=y, maze = oldParticle.maze, noisy=False,
                                heading=oldParticle.heading, weight=w, sensor_limit = oldParticle.sensor_limit)
        else:
            newParticle = Particle(x=oldParticle.x, y=oldParticle.y, maze=oldParticle.maze, noisy=True,
                                heading=oldParticle.heading, weight=oldParticle.weight, sensor_limit=oldParticle.sensor_limit)
        return newParticle

    def __systematic_resample(self, weights):
        """ Performs the systemic resampling algorithm used by particle filters.

        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.

        Parameters
        ----------
        weights : list-like of float
       	    list of weights as floats

        Returns
        -------

        indexes : ndarray of ints
            array of indexes into the weights defining the resample. i.e. the
            index of the zeroth resample is indexes[0], etc.
        """
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.random() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N and j < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    
    def __multinomial_resample(self, weights):
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.
        return np.searchsorted(cumulative_sum, np.random.random(size=self.num_particles))

    def __stratified_resample(self, weights):
        """ Performs the stratified resampling algorithm used by particle filters.

        This algorithms aims to make selections relatively uniformly across the
        particles. It divides the cumulative sum of the weights into N equal
        divisions, and then selects one particle randomly from each division. This
        guarantees that each sample is between 0 and 2/N apart.

        Parameters
        ----------
        weights : list-like of float
            list of weights as floats

        Returns
        -------

        indexes : ndarray of ints
            array of indexes into the weights defining the resample. i.e. the
            index of the zeroth resample is indexes[0], etc.
        """

        N = len(weights)
        # make N subdivisions, and chose a random position within each one
        positions = (np.random.random(N) + range(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        ## TODO #####

        idx_new = self.__stratified_resample([x.weight for x in self.particles])
        self.particles = [self.makeParticle(i) for i in idx_new]

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
        """
        ## TODO #####

        if len(self.control) == 0: return # TODO
        vars = [self.bob.x, self.bob.y, self.bob.heading]
    
        for particle in self.particles:
            # r = ode(vehicle_dynamics).set_integrator('lsoda', method='bdf')
            # val = [particle.x, particle.y, particle.heading]
            # for control in self.control[self.last_integration:]:
            # 	r.set_f_params(vars, *control)
            # 	r.set_initial_value(val)

            # 	val = r.integrate(r.t + 0.01)
            
            # particle.x = val[0].astype(np.float64)
            # particle.y = val[1].astype(np.float64)
            # particle.heading = val[2].astype(np.float64)

            for v, delta in self.control[self.last_integration:]:
                theta = particle.heading
            	particle.x += 0.01 * v * np.cos(theta)
                particle.y += 0.01 * v * np.sin(theta)
		particle.heading += 0.01 * delta

        self.last_integration = len(self.control)
        
        # TODO: optimize

    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        while True:
            ## TODO #####
            # Finish this function to have the particle filter running

            # sampleMotionModel(p)
            self.particleMotionModel()

            # Read sensor msg

            # reading = vehicle_read_sensor()
            reading = self.bob.read_sensor()

            # updateWeight(p, reading)
            self.updateWeight(reading)

            # p = resampleParticle(p)
            self.resampleParticle()
        
            
            # Display robot and particles on map 
            self.world.show_particles(particles = self.particles, show_frequency = 10)
            self.world.show_robot(robot = self.bob)
            [est_x,est_y] = self.world.show_estimated_location(particles = self.particles)
            self.world.clear_objects()

            ###############
