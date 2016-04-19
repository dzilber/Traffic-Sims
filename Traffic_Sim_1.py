import matplotlib.pyplot as plt
import matplotlib.animation as anm
import pandas as pd
import numpy as np
plt.style.use('ggplot')


'''Traffic Sim 1
Bad design, chose linear but polar should be easier



Models single row of vehicles moving through road (loop or single lane)

Basic vars:
X(t) = Position at time t for x1, x2, ..., xn
D(t) = distance between xi and xi+1 
V(t) = velocity at time t

parameters:
am = max acceleration
ad = max deceleration
Vm = max velocity
Tr = reaction time

time step = dt = min(1, min_i { d_i/(V_i- V_i+1)})
Given dt, X+ = X + V*dt
Di = Xi+1-Xi
 
Velocity & Psychology:
Basic idea, keep distance < velocity * stopping time
stopping time = Tr + Tslow
Tslow = V/ad
V = min(d/(Tr + V/ad), Vm)

'''

TRACK_LENGTH = 300
VELOCITY_MAX = 30
ACCELERATION_MAX = 10
ACCELERATION_MIN = -1
TIME_REACTION = [10, 1, 2, 1]



def get_next_step(X,V,D):
    intercept_time = [D[i]/(max(.000001, V[i])) for i in range(len(V))]
    # going until intercept time unnecessary, go to half?
    dt = min(1./30, min(intercept_time)/2)
    return dt

def update_Pos(dt, X, V):
    # move forward according to speed and time
    X_n = [X[i]+V[i]*dt for i in range(len(X))]
    # teleport back according to track length
    X_n = [x%TRACK_LENGTH for x in X_n]
    # must resort using velocity to maintain integrity
    paired = [i for i in zip(X_n, V)]
    paired.sort()
    X_n, V = zip(*paired)
    # Update distances
    D = [X_n[i+1]-X_n[i] for i in range(len(X_n)-1)] + [TRACK_LENGTH-X_n[-1]+X_n[0]]
    return X_n, V, D


def velocity_change_rule(max_vel, current_vel):
    new_velocity = current_vel
    if max_vel>current_vel:
        new_velocity += min(ACCELERATION_MAX, max_vel-current_vel)

    elif max_vel<current_vel:
        # take max because accel and delta <0
        new_velocity += max(ACCELERATION_MIN, max_vel-current_vel)

    # if equal, no change
    
    if new_velocity > VELOCITY_MAX:
        new_velocity = VELOCITY_MAX

    # randomly set velocity to 0
    if np.random.rand()>.999:
        new_velocity = 100
    return max(new_velocity, 0)



def update_velocity(X,V,D):
    indices = range(len(V))
    # get velocity difference for relative speed
    dv = [V[i]-V[(i+1)%len(V)] for i in indices]

    # avoid crash:  t_decel = V/ad
    # Assume front car slows at same rate:  dV/ad

    # if relative speed < 0, cannot catch up, velocity can increase
    denom = [TIME_REACTION[i] + dv[i]/abs(ACCELERATION_MIN) for i in indices]

    max_velocity = [D[i]/max(.01,denom[i]) for i in indices]
    #  if max_velocity>Vi, accelerate by min(am, max_v-Vi)
    # if max_v = Vi, same
    # if max_v < Vi, decel by min (ad, Vi-max_V)

    V_n = [velocity_change_rule(mv, v) for mv, v in zip(max_velocity, V)]
    return V_n
        



def take_step(X,V,D):
    dt = get_next_step(X,V,D)
    print('Time Step:',dt)
    X, V, D = update_Pos(dt, X, V)
    print('Pos',X)
    print('Velocity',V)
    # print('Distance',D)

    V = update_velocity(X, V, D)
    print('New Velocity',V)

    return X,V,D, dt



def convert_polar(xv):
    theta = [x*(np.pi * 2)/TRACK_LENGTH for x in xv]
    r = TRACK_LENGTH / (np.pi * 2) 
    yn = [r*np.sin(t) for t in theta]
    xn = [r*np.cos(t) for t in theta]
    return xn,yn





# Pt 1, vars
X = (0, 5, 8, 99)
V = [10, 0, 1, 10]#, 10, 5]
D = [X[i+1]-X[i] for i in range(len(X)-1)] + [TRACK_LENGTH-X[-1]+X[0]]

total_time = 0
hist = [(total_time, X,V,D)]
while total_time<90:
    X,V,D,dt = take_step(X,V,D)
    total_time+=dt
    hist.append((total_time, X,V,D))

tl, xl, vl, dl = zip(*hist)
# x,y, z = zip(*xl)
# plt.scatter(tl,x)
# plt.scatter(tl,y)
# plt.scatter(tl,z)
# plt.show()

# tln, xln = interpolate_data(tl, xl)
# bleh


time_gen = zip(tl, xl)
fig, ax = plt.subplots()
ax.set_xlim([-TRACK_LENGTH/5,TRACK_LENGTH/5])
ax.set_ylim([-TRACK_LENGTH/5,TRACK_LENGTH/5])
cars, = ax.plot([],[],'bo', ms=15)

def run(input_state):
    t, xv = input_state

    x,y = convert_polar(xv)

    cars.set_data(x,y)
    # [ax.scatter(0,p) for p in xv]
    


ani = anm.FuncAnimation(fig, run, iter(time_gen), blit=False, interval = 10,
    repeat=False)

plt.show()