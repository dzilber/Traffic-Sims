
''' 
Sim V3

Using simpy, spawn car processes that meet at central intersection

'''

import simpy as sp
import scipy.spatial.distance as spd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import itertools
from collections import namedtuple
from operator import itemgetter


'''Model Parameters'''
bound = [0, 10, 0, 1 ]
NUM_CARS = 20
num_drivers = 5
CAR_LENGTH = .2


Driver_obj = namedtuple('Driver', ['ID','Vx', 'Vy', 'Reaction_Time'])
drivers = sp.Store(env, capacity = num_drivers)
for i in range(num_drivers):
    driver = Driver_obj('D'+str(i), np.random.rand(), 0, np.random.rand()/10) # speed x,y, reaction etc?
    drivers.put(driver)


'''Time delta for sim'''
dt = 1./30


'''
Map of world
Consists of roads and intersections, each which has bounds
'''
# road = x0, y0, x1, y1
r1 = [0, 1, 5, 1]
r2 = [5, 1, 5, 6]
r3 = [5, 1, 10, 1]
r4 = [5, 6, 10, 6]


intersection = [[]]



'''
Car Process:
    - Request resource (num cars allowed)
    - fixed starting point
    - generate path
    - car steps = update position for time
    - 
'''
class Car():
    def __init__(self, car_id, pos, v, driver_resource, env, path):
        self.env = env
        self.car_id = car_id
        self.driver_pool = driver_resource
        self.xy = pos # [x, y]
        self.vxy = v # [vx, vy]

        # instead of boundary, have path
        
        self.history = []
        self.reaction_time = 1 # default
        self.active = False

    #  Equivalent to java toString()
    def __repr__(self):
        return 'Car_'+str(self.car_id)

    def start_drive_process(self):
        # request driver
        # with self.driver_pool.request() as driver:
            # yield driver
    
        driver = yield self.driver_pool.get()
        self.vxy = np.array((driver.Vx,driver.Vy))
        self.reaction_time = driver.Reaction_Time
        self.active = True
        self.driver_id = driver.ID
        # self.pos=np.array([0.,0.])
        yield self.env.process(self.traverse())
        self.driver_pool.put(driver)

        #end

    def traverse_road(self):
        in_boundary = True
        while in_boundary:
            yield self.env.timeout(dt)
            # progress by timestamp 
            in_boundary = self.update_pos(dt)
            # update global status for other cars to react
            self.active = in_boundary
            # logic to control speed
            self.update_vel()
            # save local record
            self.record_state()



    def update_pos(self, dt):

        self.xy += dt * self.vxy

        # check if boundary hit
        left_x = self.xy[0]<self.boundary[0]
        right_x = self.xy[0]>self.boundary[1]
        low_y = self.xy[1]<self.boundary[2]
        hi_y = self.xy[1]>self.boundary[3]

        # remove cars that left space?
        return -np.array((left_x, right_x, low_y, hi_y)).any()

    def update_vel(self):
        # get positions using global var current_positions
        # current_positions: [is_active, car num, x,y, vx, vy]
        global car_list

        active_cars = [c for c in car_list if c.active]
        other_active_cars = [c for c in active_cars if c.car_id != self.car_id]
        # print(other_active_cars)
        

        active_xy = [ac.xy for ac in other_active_cars if ac.xy[0]>self.xy[0]]
        print(active_xy)
        # sort , take lowest value
        #Sort
        active_xy = sorted(active_xy, key=itemgetter(0))

        if len(active_xy)>0:
            delta = max(0,active_xy[0][0]-self.xy[0]-CAR_LENGTH)
            # if to close, given current speed: reduce
            if self.vxy[0] > delta/self.reaction_time:
                self.vxy[0] = delta/self.reaction_time
            elif self.vxy[0] < delta/self.reaction_time:
                self.vxy[0]+=.01
        else:
            self.vxy[0]+=.01


        # bloop

        
        

    def record_state(self):
        # create list of arrays, concatenate
        state = [self.car_id] + list(self.xy) + list(self.vxy) + [self.env.now]
        # save state to local history
        # to be called after sim is finished for visualization 
        self.history.append(state)

    def set_global_status(self, in_boundary):
        global current_positions
        #update global position data to show activity, for logic
        state = [in_boundary, self.car_id] + list(self.xy) + list(self.vxy)
        current_positions[int(self.car_id),:] = np.array(state)


class Intersection():
    def __init__(self, signal_logic, **roads):
        self.roads = roads

        self.state_per_road = []
        self.logic = signal_logic


    '''
    Example state:
         R1: 0
        \___/
 R4:1   |___|  R2 :1
        /   \ 
         R3:0


    states:  
    0 = Red
    1 = Green

    Future:
    11 = Green and left turn
    01 = left turn only?
    '''
    def light_process(self):
        while True:
            yield env.timeout(dt)
            self.apply_logic()

    def apply_logic(self):
        self.state_per_road = self.logic(self.roads, current_state)


def simple_alternating_logic(roads, current_signal=None):
    
    if current_state is not None:
        return {key:val==False for key,val in current_signal.items()}

    # get road orientation
    road1
    # assign


def rand_init(i):
    pos = (0,0)
    v = (np.random.rand(),0)
    # create a path?
    return str(i), pos, v, drivers, env, bound






# create all car 'jobs'
# Car(name, pos, v, driver_resource, env, boundary)
# pos = np.array([0.,0.])
v = np.array([.5,0])
car_list = [Car(i, np.array((np.random.rand(),0)), v, drivers, env, bound) for i in range(NUM_CARS)]
current_positions =np.array([np.concatenate((np.array([False, c.car_id]), c.xy, c.vxy)) for c in car_list])
print(current_positions)
# start all jobs
[env.process(c.start_drive_process()) for c in car_list]

env.run()
data = [c.history for c in car_list]
# print(data)

data = list(itertools.chain(*data))



df = pd.DataFrame(data, columns = ['ID', 'X', 'Y','Vx','Vy','Time'])
pvt_X = pd.pivot_table(df, index = 'Time', columns = 'ID', values = 'X')
pvt_Y = pd.pivot_table(df, index = 'Time', columns = 'ID', values = 'Y')
# print(pvt)
# input_data = df[['X','Y']].values
input_data_X = pvt_X.values
input_data_Y = pvt_Y.values
input_data = [r for r in zip(input_data_X, input_data_Y)]
print(input_data_X)
print(input_data_Y)
# blub




'''Animation Portion'''

fig, ax = plt.subplots()
ax.set_xlim(bound[:2])
ax.set_ylim([-.4,.4])
position_plot, = ax.plot([], [], 'o', ms=10)
clist =plt.get_cmap('jet')(np.arange(0,1,1./NUM_CARS)) 
patches = [plt.Rectangle((0,0),CAR_LENGTH,.02, fc=clist[i]) for i in range(NUM_CARS)]
[ax.add_patch(p) for p in patches]

def animate(data):
    x,y = data

    for i in range(len(x)):
        patches[i].set_xy((x[i],y[i]))

    # position_plot.set_data(x,[0]*len(x))



'''Animate Cars'''
ani = anm.FuncAnimation(fig, animate, input_data, blit=False, interval=10,
    repeat=False)

plt.show()




