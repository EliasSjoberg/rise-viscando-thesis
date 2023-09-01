import glob
import os
import sys
import h5py
import numpy as np
import carla 
import time
from queue import Queue
import array
import random
import cv2
import shutil
import re
import csv



from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from simulation_methods import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

    


def fetch_values(pop_temp):
    """
    Method on a form that deaper likes that returns the population
    """
    ret_values = pop_temp.pop(0)       
    return ret_values

def fetch_errors(avg_errors, individual):
    """
    Method on a form that deaper likes that returns the errors (fitness values)
    """
    return avg_errors.pop(0)

def get_param_values(batch_idx, pop_size, delta_timestamps, current_gen, nbr_actors, gt_path):
    """
    Reads the parameter values saved from the last generation and returns them in pop: a list containing the values.
    The values are ordered so that the starting times come first, then any potential colors and lastly the weather parameters.
    Thus, pop[0][0] is the starting time for one actor for the first individual, and pop[0][-1] is the cloudiness value for the first individual.

    Args:
    batch_idx (int): Determines which of the vehicle_tracks/model combination to simulate. Ranges from 0 to 3.
    pop_size (int): Population size.
    delta_timestamps (float): "Long frames". The amount of time between capturing of images and also the time between each waypoint in the trajectories.
    current_gen (int): The current generation.
    nbr_actors (int): The number of actors (cars, bikes, walkers) participating in the simulation.
    gt_path (str): Path to ground truth values.
    """
    #Have assumed fixed speeds

    params_path = gt_path + f'/input_params_gen{current_gen-1}/batch_{batch_idx+1}.txt' #Path to color and weather parameter values
    ts_path = gt_path + f'/ground_truth_gen{current_gen-1}_{batch_idx}' #Path to ground truth positions, speeds, classifications and timestamp values

    pop = []
    ts_save = []
    color_save = []
    weather_save = []

    for ind_idx in range(pop_size): #We start with fetching the timestamp values
        temp = [] #List containing the starting timestamps (in "long frames") for each actor in the scenario
        filegt = open(ts_path + f'/ground_truth_{ind_idx}.csv')
        csvreadergt = csv.reader(filegt, delimiter=';')
        headergt = next(csvreadergt)
        row = next(csvreadergt)

        while True:
            try:
                row_id = row[0]
                temp.append(np.rint((float(row[1])/delta_timestamps)))
                while row[0] == row_id: #We are only interested in the starting timestamp, so we just skip the rest
                    row = next(csvreadergt)
                
            except StopIteration:
                break
        ts_save.append(temp)
            
            
    
    with open(params_path,'r') as f: #Next we fetch the color and weather values
        for line in f:
            color_temp = []
            weather_temp = []
            color_match = re.findall("r: \d+   g: \d+   b: \d+", line)
            for m in color_match:
                digits = re.findall("\d+", m)
                color_temp.append(int(digits[0]))
                color_temp.append(int(digits[1]))
                color_temp.append(int(digits[2]))

            weather_match = re.findall("az: \d+\.\d+   cl: \d+\.\d+", line)
            for m in weather_match:
                nbrs = re.findall("\d+\.\d+", m)
                for nbr in nbrs:
                    weather_temp.append(float(nbr))

            color_save.append(color_temp)
            weather_save.append(weather_temp)

    for ind_idx in range(pop_size):
        pop.append(ts_save[ind_idx])
        pop[ind_idx] += color_save[ind_idx]
        pop[ind_idx] += weather_save[ind_idx]

    return pop




def main():
    print('Setting up simulator configurations...')

    #Paths:
    op_path = 'C:/Users/elias/outputs_viscando' #Path to OTUS3D estimations
    gt_path = 'C:/Users/elias/ground_truth' #Path to ground truth
    h5FilePath = 'C:/Users/eliassj/Desktop/trajectories/main_trajectories.h5' #Path to the trajectories used

    #Parameters for coordinate change
    transx = 11 #To change between Viscando coordinates and CARLA coordinates.
    transy = -9

    model = 'nsga2' #Should be 'nsga2' or 'baseline'. Note that if current_gen == 0, then model cannot be nsga2 
    current_gen = 3
    batch_idx = 3 #0-3. Determines which of the vehicle_tracks/model combination to simulate
    #0 refers to car trajectories and the baseline model, 1 to car trajectories and the NSGA-II model,
    #2 to bike trajectories and the baseline model and 3 to bike trajectories and the NSGA-II model
    
    nbr_generations = 1 #Can only do one generation before sending the results to Viscando so should always be 1
    pop_size = 20 #Should be divisible by 4 according to instructions from deap.
    crossover_prob = 0.9
    pop = array.array('I',[]) 
   
    nbr_vel_points = 3 #Number of points in which to vary velocities (only when velocities are set as variable)
    
    if batch_idx == 0 or batch_idx == 1:
        vehicle_tracks = np.array([1, 1, 1, 0, 0, 0, 0]) #Nbr vehicles for each of the seven possible tracks
        #The ones set to 1 were the car trajectories used in the report

    else:
        vehicle_tracks = np.array([0, 0, 0, 0, 1, 1, 1]) #The ones set to 1 were the bike trajectories used in the report
    
    walker_tracks = np.array([1,0,1,0,0,1,1]) #The ones set to 1 were the walker trajectories used in the report
    static_tracks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #Should have length equal to vehicle_tracks and walker_tracks combined.
    #Static tracks are constant across generations. Note that we can only have one of each track to be static.
    #No static tracks were used in the report
    delta_timestamps = 0.05 #"Long frames". The amount of time between capturing of images and also the time between each waypoint in the trajectories.
    delta_seconds = delta_timestamps/3 #"Short frames". Amount of simulated time per frame. Must be 1, 1/2, 1/3 etc of delta_timestamps!

    t_max = 10 #max time (s) where creation of actor is allowed.
    v_max = 12 #max speed allowed for cars (m/s) (only when varying velocities)
    v_max_w = 1.9 #max speed allowed for walkers (m/s)
    v_max_b = 6 #max speed allowed for bikes (m/s)
    azimuth_min = 25 #If changed, also needs to be changed in generate_values in simulation_methods.
    azimuth_max = 265 - (azimuth_min - 15)
    nbr_frames_ts = int(round(t_max/delta_timestamps)) #Number of "long frames" where creation of actor is allowed.
    nbr_tracks = np.sum(vehicle_tracks)
    nbr_car_tracks = np.sum(vehicle_tracks[:3])
    nbr_bike_tracks = np.sum(vehicle_tracks[3:7])
    nbr_walker_tracks = np.sum(walker_tracks)
    tracks = np.concatenate((vehicle_tracks,walker_tracks))
    total_nbr_tracks = nbr_tracks + nbr_walker_tracks + np.sum(static_tracks)

    bool_dict = dict() #Determines which parameters should be variable across generations. True means that the parameter is variable
    bool_dict['timestamps'] = True
    bool_dict['vels'] = False #Note: Setting this to True has not been properly tested and might yield unsatisfactory results
    bool_dict['colors'] = True
    bool_dict['clouds'] = True
    bool_dict['precip'] = False
    bool_dict['wind'] = False
    bool_dict['sun_angle'] = True
    bool_dict['fog'] = False
    bool_dict['wetness'] = False

    weather_dict = { #Default values
        'cloudiness': 40,
        'precipitation': 0,
        'wind_intensity': 15,
        'sun_azimuth_angle': 140,
        'sun_altitude_angle': 50,
        'fog_density': 10,
        'fog_distance': 75,
        'wetness': 0,
        'fog_falloff': 1.0,
        'scattering_intensity': 0
    }

    #Camera coordinates
    Mleft = np.array([[ 7.20075210e-01,  1.19859060e-01,  6.83465798e-01,  1.75678427e+01],
                     [ 1.69352137e-01, -9.85539755e-01, -5.58975566e-03,  2.81069847e+00],
                     [ 6.72912732e-01,  1.19771438e-01, -7.29961134e-01,  7.38748717e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    Minvleft = np.linalg.inv(Mleft)

    Mright = np.array([[ 7.20649715e-01,  1.28220763e-01,  6.81339434e-01,  1.75605976e+01],
                     [ 1.72248823e-01, -9.85048311e-01,  3.18869100e-03,  3.33110925e+00],
                     [ 6.71561115e-01,  1.15061986e-01, -7.31960660e-01,  7.35169228e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    Minvright = np.linalg.inv(Mright)

    #Camera coordinates in the simulation.
    coordsleft = Minvleft[0:3,3]
    coordsright = Minvright[0:3,3]
    coordsleft[0] += transx - 1
    coordsright[0] += transx - 1
    coordsleft[1] += transy
    coordsright[1] += transy
    coordsleft[2] = -coordsleft[2]
    coordsright[2] = -coordsright[2]

    #Note that these are in radians
    pitchleft = -np.arcsin(Minvleft[2,0])
    pitchright = -np.arcsin(Minvright[2,0])
    yawleft = np.arcsin(Minvleft[2,1]/np.cos(pitchleft))
    yawright = np.arcsin(Minvright[2,1]/np.cos(pitchright))
    rolleft = np.arcsin(Minvleft[1,0]/np.cos(pitchleft))
    rollright = np.arcsin(Minvright[1,0]/np.cos(pitchright))
   
    try:
        #Set up CARLA
        client = carla.Client('127.0.0.1',2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta_seconds
        world.apply_settings(settings)
        

        #Fetch input data and put into dicts
        track_dict = dict() #Keeping track of the indices of the trajectories
        coord_dict = dict() #Positional coordinates of the trajectories
        speed_dict = dict() #Speed at waypoints in the trajectories
        color_dict = dict() #Colors of the cars in the simulation
        starting_timestamps = np.zeros(total_nbr_tracks) #starting times of the actors in the simulation
        object_classes = np.zeros(total_nbr_tracks) #Keeping track of whether the actor is a car, bicycle or walker
        start_idx = 0 #How many of the starting frames to remove
        h5File = h5py.File(h5FilePath, 'r')
       
        idx1 = 0
        idx2 = 0
       
        while idx1 < len(tracks):
            if tracks[idx1] > 0:
                #Matrix used in trajectory coordinate change:
                theta = h5File['Tracks'][f"{idx1}"]['States']['Theta'][0]
                transMatrix = np.matrix(np.identity(3))
                transMatrix[0,2] = transx
                transMatrix[1,2] = transy
                transMatrix[0,0] = np.cos(np.radians(theta))
                transMatrix[0,1] = -np.sin(np.radians(theta))
                transMatrix[1,0] = np.sin(np.radians(theta))
                transMatrix[1,1] = np.cos(np.radians(theta))
                
                xs = h5File['Tracks'][f"{idx1}"]['States']['X_local_coordinate_system'][start_idx:]
                ys = h5File['Tracks'][f"{idx1}"]['States']['Y_local_coordinate_system'][start_idx:]
                coords = np.stack((xs,ys,np.ones(len(xs))))
                coords = np.matmul(transMatrix,coords) #Do coordinate change
                coord_dict[idx2] = coords
               
                if bool_dict['vels'] == False:
                    speed_dict[idx2] = h5File['Tracks'][f"{idx1}"]['States']['Velocity_ms'][start_idx:] #Default speed values

                if bool_dict['timestamps'] == False:
                    starting_timestamps[idx2] = h5File['Tracks'][f"{idx1}"]['States']['Timestamps_UNIX'][start_idx] #Default timestamps
                
                object_classes[idx2] = h5File['Tracks'][f"{idx1}"]['States']['Object_class'][0]
                if object_classes[idx2] == 18: #If the actor is a car
                    color_dict[idx2] = [17,37,103] #Default color values.

                track_dict[idx2] = idx1
                idx2 += 1
                tracks[idx1] -= 1
            else:
                idx1 += 1

        
        #Note that there can only be one of each of the static tracks since the timestamps are the same.
        for idx in range(len(static_tracks)):
            if static_tracks[idx] > 0:
                theta = h5File['Tracks'][f"{idx}"]['States']['Theta'][0]
                transMatrix = np.matrix(np.identity(3))
                transMatrix[0,2] = transx
                transMatrix[1,2] = transy
                transMatrix[0,0] = np.cos(np.radians(theta))
                transMatrix[0,1] = -np.sin(np.radians(theta))
                transMatrix[1,0] = np.sin(np.radians(theta))
                transMatrix[1,1] = np.cos(np.radians(theta))
                
                xs = h5File['Tracks'][f"{idx}"]['States']['X_local_coordinate_system'][start_idx:]
                ys = h5File['Tracks'][f"{idx}"]['States']['Y_local_coordinate_system'][start_idx:]            
                coords = np.stack((xs,ys,np.ones(len(xs))))
                coords = np.matmul(transMatrix,coords) #Do coordinate change
                coord_dict[idx2] = coords
               
                speed_dict[idx2] = h5File['Tracks'][f"{idx}"]['States']['Velocity_ms'][start_idx:]

                starting_timestamps[idx2] = h5File['Tracks'][f"{idx}"]['States']['Timestamps_UNIX'][start_idx]

                object_classes[idx2] = h5File['Tracks'][f"{idx}"]['States']['Object_class'][0]
                if object_classes[idx2] == 18:
                    color_dict[idx2] = [17,37,103] #Default color values.

                idx2 += 1
                                   
               
        #Setup RGB camera
        sensor_list = []
        sensor_queue = Queue()
        
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(2048))
        cam_bp.set_attribute("image_size_y",str(1536))
        cam_bp.set_attribute("fov",str(101.5))
        cam_bp.set_attribute("sensor_tick",f"{delta_timestamps}")
        
        cam_location_left = carla.Location(coordsleft[0],coordsleft[1],coordsleft[2])
        cam_rotation_left = carla.Rotation(np.degrees(pitchleft)-4,np.degrees(yawleft)+7,np.degrees(rolleft)-6)
        cam_transform_left = carla.Transform(cam_location_left,cam_rotation_left)
        rgb_cam_left = world.spawn_actor(cam_bp,cam_transform_left)
        cam_location_right = carla.Location(coordsright[0],coordsright[1],coordsright[2])
        cam_rotation_right = carla.Rotation(np.degrees(pitchright)-4,np.degrees(yawright)+7,np.degrees(rollright)-6)
        cam_transform_right = carla.Transform(cam_location_right,cam_rotation_right)
        rgb_cam_right = world.spawn_actor(cam_bp,cam_transform_right)
        
        rgb_cam_left.listen(lambda data: sensor_callback(data, sensor_queue, 0))
        rgb_cam_right.listen(lambda data: sensor_callback(data, sensor_queue, 1))
        
        sensor_list.append(rgb_cam_left)
        sensor_list.append(rgb_cam_right)
        
        starting_frames_ts = np.rint(starting_timestamps/delta_timestamps) #Starting frames in long frames
        
        max_frames = np.zeros(len(tracks), dtype = np.uint32) #The number of waypoints or "long frames" in each trajectory
        for idx in range (len(tracks)):
            xs = h5File['Tracks'][f"{idx}"]['States']['X_local_coordinate_system'][start_idx:]
            max_frames[idx] = len(xs)

        if model == 'nsga2': #This block is about fetching past values and evaluating so that the nsga-ii algorithm can be used
            nbr_elements = [bool_dict['timestamps']*(nbr_tracks+nbr_walker_tracks),bool_dict['vels']*nbr_car_tracks*nbr_vel_points,
                            bool_dict['vels']*nbr_bike_tracks*nbr_vel_points,bool_dict['vels']*nbr_walker_tracks*nbr_vel_points,
                            bool_dict['colors']*len(color_dict)*3,bool_dict['sun_angle'],bool_dict['clouds'],bool_dict['precip'],bool_dict['wind'],
                            bool_dict['fog'],bool_dict['wetness']]

            BOUND_LOW = [0]*nbr_elements[0] + [0.5]*nbr_elements[1] + [0.5]*nbr_elements[2] + [0.3]*nbr_elements[3] + [0]*nbr_elements[4] + [azimuth_min]*nbr_elements[5] + \
                        [0]*nbr_elements[6] + [0]*nbr_elements[7] + [0]*nbr_elements[8] + [0]*nbr_elements[9] + [0]*nbr_elements[10]

            #Lower bounds for variable parameters. Minimum speeds are arbitrarily set to 0.5m/s for cars/bikes and 0.3m/s for walkers.
            #If any of these lower bounds are changed, they also need to be changed in generate_values in simulation_methods.
            #The reason is to keep the code simpler.
            
            BOUND_UP = [nbr_frames_ts]*nbr_elements[0] + [v_max]*nbr_elements[1] + [v_max_b]*nbr_elements[2] + [v_max_w]*nbr_elements[3] + [255]*nbr_elements[4] + \
            [azimuth_max]*nbr_elements[5] + [100]*nbr_elements[6] + [100]*nbr_elements[7] + [100]*nbr_elements[8] + [100]*nbr_elements[9] + [100]*nbr_elements[10]        
           
            creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0)) #Assuming three objectives
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax) #Create individual

            toolbox = base.Toolbox()

            pop_temp = get_param_values(batch_idx, pop_size, delta_timestamps, current_gen, total_nbr_tracks, gt_path) #The population from the last generation

            toolbox.register("attr_float", fetch_values, pop_temp)
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual) #Needed in order to have the population on the "correct" form
            #so that deaper methods can be used.

            (avg_errors, dist_results, speed_results, perc_misclass, step_perc) = compute_errors(batch_idx, total_nbr_tracks, pop_size, current_gen, False, op_path, gt_path)
            #returns a list containing 20 errors.
            """
            #Compute the mean fitness/errors across all individuals in the population
            mean_fitness_dist = np.zeros(7)
            mean_fitness_speed = np.zeros(7)
            mean_fitness_perc = np.zeros(7)

            for idx1 in range(len(dist_results)):
                mean_fitness_dist += dist_results[idx1]
                mean_fitness_speed += speed_results[idx1]
                mean_fitness_perc += perc_misclass[idx1]

            mean_fitness_dist = mean_fitness_dist/len(dist_results)
            mean_fitness_speed = mean_fitness_speed/len(dist_results)
            mean_fitness_perc = mean_fitness_perc/len(dist_results)

            print(mean_fitness_dist)
            print(mean_fitness_speed)
            print(mean_fitness_perc)
            print(np.mean(mean_fitness_dist))
            print(np.mean(mean_fitness_speed))
            print(np.mean(mean_fitness_perc))
            """

            toolbox.register("evaluate", fetch_errors, avg_errors)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0) #Lower eta makes the child more different from the parent.
            toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/len(BOUND_LOW))
            toolbox.register("select", tools.selNSGA2)

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"

            if current_gen == 1:
                pop = toolbox.population(n=pop_size)

                # Evaluate the individuals with an invalid fitness (all individuals at this stage)
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                print("last gen population hypervolume is %f" % hypervolume(pop))

                # This is just to assign the crowding distance to the individuals
                # no actual selection is done
                pop = toolbox.select(pop, len(pop))
                
                record = stats.compile(pop)
                #logbook.record(gen=0, evals=len(invalid_ind), **record)
                #print(logbook.stream)

            else:
                #In the nsga-ii algorithm, we need the parameter values from two generations back when doing selection.
                last_gen_temp = get_param_values(batch_idx, pop_size, delta_timestamps, current_gen-1, total_nbr_tracks, gt_path)
                toolbox.register("attr_float_last_gen", fetch_values, last_gen_temp) 
                toolbox.register("individual_last_gen", tools.initIterate, creator.Individual, toolbox.attr_float_last_gen)
                toolbox.register("population_last_gen", tools.initRepeat, list, toolbox.individual_last_gen)

                (avg_errors_last_gen, dist_results, speed_results, perc_misclass, step_perc) = compute_errors(batch_idx, total_nbr_tracks, pop_size, current_gen-1, False, op_path, gt_path)
                toolbox.register("evaluate_last_gen", fetch_errors, avg_errors_last_gen)
                
                pop = toolbox.population_last_gen(n=pop_size)
                offspring = toolbox.population(n=pop_size)

                invalid_ind_pop = [ind for ind in pop if not ind.fitness.valid]
                fitnesses_pop = toolbox.map(toolbox.evaluate_last_gen, invalid_ind_pop)
                for ind, fit in zip(invalid_ind_pop, fitnesses_pop):
                    ind.fitness.values = fit
                    
                invalid_ind_offspr = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses_offspr = toolbox.map(toolbox.evaluate, invalid_ind_offspr)
                for ind, fit in zip(invalid_ind_offspr, fitnesses_offspr):
                    ind.fitness.values = fit

                print("last gen population hypervolume is %f" % hypervolume(offspring))
                # Select the next generation population among pop and offspring:
                pop = toolbox.select(pop + offspring, pop_size)
                print("next gen population hypervolume is %f" % hypervolume(pop))
                record = stats.compile(pop)
                #logbook.record(gen=current_gen, evals=len(invalid_ind), **record)
                #print(logbook.stream)


               
        world.tick() #Tick to spawn in the cameras.
        print('Starting the loop...')
       
        for gen_nbr in range(nbr_generations):
            if model == 'nsga2':
                # Vary the population
                offspring = tools.selTournamentDCD(pop, len(pop))
                offspring = [toolbox.clone(ind) for ind in offspring] #Need this since otherwise we make changes to the individuals in pop.
                ctr = 0
                for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() <= crossover_prob:
                        toolbox.mate(ind1, ind2)
                    toolbox.mutate(ind1)
                    toolbox.mutate(ind2)
                    del ind1.fitness.values, ind2.fitness.values
                    ctr += 1

                ind_nbr = 0
                for ind in offspring:
                    (starting_frames_ts,speed_dict,color_dict,weather_dict) = generate_values(bool_dict, nbr_car_tracks, nbr_bike_tracks, nbr_walker_tracks, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames_ts, weather_dict, speed_dict, track_dict, color_dict, starting_frames_ts, max_frames, ind)
                    #generate_values fetches values from ind and puts them into dicts
                    starting_frames = np.rint(starting_frames_ts*delta_timestamps/delta_seconds) #Change from long to short frames

                    save_params(batch_idx, ind_nbr, color_dict, weather_dict) #Save ground truth colors and weathers. The positions and speeds are saved later
    
                    weather = set_weather_params(weather_dict)
                    world.set_weather(weather)

                    run_simulation(ind_nbr, batch_idx, total_nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library)

                    save_video(batch_idx, ind_nbr)
                    
                    ind_nbr += 1


            else: #The baseline model
                for ind_nbr in range(pop_size):
                    (starting_frames_ts,speed_dict,color_dict,weather_dict) = generate_values(bool_dict, nbr_car_tracks, nbr_bike_tracks, nbr_walker_tracks, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames_ts, weather_dict, speed_dict, track_dict, color_dict, starting_frames_ts, max_frames, pop)
                    #Since len(ind)==0, generate_values generates random values
                    starting_frames = np.rint(starting_frames_ts*delta_timestamps/delta_seconds) #Change from long to short frames

                    save_params(batch_idx, ind_nbr, color_dict, weather_dict)

                    weather = set_weather_params(weather_dict)
                    world.set_weather(weather)

                    run_simulation(ind_nbr, batch_idx, total_nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library)
                    save_video(batch_idx, ind_nbr)


        print('Done with loop')

    
    finally:
        world.apply_settings(original_settings)
        for s in sensor_list:
            s.destroy()
       
    
        

if __name__ == '__main__':
    try:
        start_time = time.time()
        main()
        print("--- %s seconds ---" % (time.time() - start_time))
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

