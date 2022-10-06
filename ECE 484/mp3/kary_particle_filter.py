import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode

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
        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle

        self.t = 0
        return

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)
        print("control updated", len(self.control))

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

        s = 0
        for particle in self.particles:
            reading_particle = particle.read_sensor()
            w = self.weight_gaussian_kernel(reading_particle, readings_robot)
            particle.weight = w
            s += w

        for particle in self.particles:
            particle.weight /= s


    def makeParticle(self, idx):
        oldParticle = self.particles[idx]
        newParticle = Particle(x = oldParticle.x, y = oldParticle.y, maze = oldParticle.maze, 
                                heading=oldParticle.heading, weight = oldParticle.weight, sensor_limit = oldParticle.sensor_limit, noisy=True)
        return newParticle

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####
        cum_w = np.cumsum([particle.weight for particle in self.particles])
        rw = np.random.uniform(0, 1, len(self.particles))
        rw.sort()

        idxs = []
        wi = 0
        i = 0
        while i < len(rw):
            w = rw[i]
            if wi >= len(cum_w): 
                particles_new.append(self.makeParticle(-1))
                i += 1
            elif w <= cum_w[wi]: 
                particles_new.append(self.makeParticle(wi))
                i += 1
            else: wi += 1

        ###############

        self.particles = particles_new

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
        """
        ## TODO #####

        # def dx(t, x, v, theta):
        #     return v * np.cos(theta)
        
        # def dy(t, y, v, theta):
        #     return v * np.sin(theta)
        
        # def dtheta(t, delta):
        #     return delta

        if len(self.control) == 0: return # TODO
        # vars = [self.bob.x, self.bob.y, self.bob.heading]

        # print("particle motion called", len(self.control))


        for particle in self.particles:
            # r = ode(vehicle_dynamics).set_integrator('zvode', method='bdf')
            # r.set_initial_value([particle.x, particle.y, particle.heading], 0)
            # vars = [self.bob.x, self.bob.y, particle.heading]

            for i in range(self.t, len(self.control)):
                
                # r.set_f_params(vars, self.control[i][0], self.control[i][1])
                # val = r.integrate(r.t + 0.01)
                
                # particle.x = val[0].astype(np.float32)
                # particle.y = val[1].astype(np.float32)
                # particle.heading = val[2].astype(np.float32)
                
                
                v = self.control[i][0]
                delta = self.control[i][1]
                theta = particle.heading
                particle.x += 0.01 * v * np.cos(theta)
                particle.y += 0.01 * v * np.sin(theta)
                particle.heading += 0.01 * delta
        
        self.t = len(self.control)
        

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
            if len(self.control) == 0: continue # TODO
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
