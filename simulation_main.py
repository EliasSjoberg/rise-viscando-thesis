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

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def main():

    #Parameters for coordinate change
    print('Setting up simulator configurations...')
    transx = 11 #To change between Viscando coordinates and CARLA coordinates.
    transy = -9 

    model = 'baseline'
    nbr_generations = 20
    pop_size = 40 #Should be divisible by 4.
    crossover_prob = 0.9
    pop = array.array('I',[]) 
   
    nbr_vel_points = 3 #Number of points in which to vary velocities.
    vehicle_tracks = np.array([0, 0, 0, 0, 1, 1, 1])
    walker_tracks = np.array([1,0,1,0,0,1,1])
    static_tracks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #Should have length equal to vehicle_tracks and walker_tracks combined.
    #Note that we can only have one of each track to be static.
    delta_timestamps = 0.05
    delta_seconds = delta_timestamps/3 #Must be 1, 1/2, 1/3 etc of delta_timestamps!
    v_max = 12 #max speed allowed (m/s)
    t_max = 10 #max time where creation of actor is allowed.
    v_max_w = 1.9
    v_max_b = 6
    azimuth_min = 25
    azimuth_max = 265 - (azimuth_min - 15)
    nbr_frames_ts = int(round(t_max/delta_timestamps)) #long frames. Only if we keep timestamps as parameter
    nbr_tracks = np.sum(vehicle_tracks)
    nbr_car_tracks = np.sum(vehicle_tracks[:3])
    nbr_bike_tracks = np.sum(vehicle_tracks[3:7])
    nbr_walker_tracks = np.sum(walker_tracks)
    tracks = np.concatenate((vehicle_tracks,walker_tracks))
    total_nbr_tracks = nbr_tracks + nbr_walker_tracks + np.sum(static_tracks)

    bool_dict = dict() #Which parameters should be dynamic across generations?
    bool_dict['timestamps'] = True #True means that the parameter is dynamic
    bool_dict['vels'] = False
    bool_dict['colors'] = True
    bool_dict['clouds'] = True
    bool_dict['precip'] = False
    bool_dict['wind'] = False
    bool_dict['sun_angle'] = True
    bool_dict['fog'] = False
    bool_dict['wetness'] = False

    weather_dict = { #Default values.
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
        client = carla.Client('127.0.0.1',2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta_seconds
        world.apply_settings(settings)

        gen_nbr = 0


        #Fetch input data
        h5FilePath = 'C:/Users/eliassj/Desktop/trajectories/main_trajectories.h5' #Fetching the trajectories.
        track_dict = dict() #Keeping track of the indices of the tracks.
        coord_dict = dict()
        speed_dict = dict()
        color_dict = dict()
        starting_timestamps = np.zeros(total_nbr_tracks)
        object_classes = np.zeros(total_nbr_tracks)
        start_idx = 0 #How many of the starting frames to remove.
        h5File = h5py.File(h5FilePath, 'r')
       
        idx1 = 0
        idx2 = 0
       
        while idx1 < len(tracks):
            if tracks[idx1] > 0:
                #Matrix used in coordinate change:
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
                    speed_dict[idx2] = h5File['Tracks'][f"{idx1}"]['States']['Velocity_ms'][start_idx:]

                if bool_dict['timestamps'] == False:
                    starting_timestamps[idx2] = h5File['Tracks'][f"{idx1}"]['States']['Timestamps_UNIX'][start_idx]

                object_classes[idx2] = h5File['Tracks'][f"{idx1}"]['States']['Object_class'][0]
                if object_classes[idx2] == 18:
                    color_dict[idx2] = [17,37,103] #Default values.

                track_dict[idx2] = idx1
                idx2 += 1
                tracks[idx1] -= 1
            else:
                idx1 += 1

        
        #Note that there can only be one of each of the static tracks since the timestamps are the same.
        for idx in range(len(tracks)):
            if static_tracks[idx] > 0:
                theta = h5File['Tracks'][f"{idx1}"]['States']['Theta'][0]
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
                    color_dict[idx2] = [17,37,103] #Default values.

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
        
        world.tick() #Tick to spawn in the cameras.

        starting_frames_ts = np.rint(starting_timestamps/delta_timestamps) #Needs to be in long frames.
        max_frames = np.zeros(len(tracks), dtype = np.uint32)

        for idx in range (len(tracks)):
            xs = h5File['Tracks'][f"{idx}"]['States']['X_local_coordinate_system'][start_idx:]
            max_frames[idx] = len(xs)

        if model == 'nsga2':
            nbr_elements = [bool_dict['timestamps']*(nbr_tracks+nbr_walker_tracks),bool_dict['vels']*nbr_car_tracks*nbr_vel_points,
                            bool_dict['vels']*nbr_bike_tracks*nbr_vel_points,bool_dict['vels']*nbr_walker_tracks*nbr_vel_points,
                            bool_dict['colors']*len(color_dict)*3,bool_dict['clouds'],bool_dict['precip'],bool_dict['wind'],bool_dict['sun_angle'],
                            bool_dict['fog'],bool_dict['wetness']]

            BOUND_LOW = [0]*nbr_elements[0] + [0.5]*nbr_elements[1] + [0.5]*nbr_elements[2] + [0.3]*nbr_elements[3] + [0]*nbr_elements[4] + [0]*nbr_elements[5] + \
                        + [0]*nbr_elements[6] + [0]*nbr_elements[7] [azimuth_min]*nbr_elements[8] + [0]*nbr_elements[9] + [0]*nbr_elements[10]

            BOUND_UP = [t_max]*nbr_elements[0] + [v_max]*nbr_elements[1] + [v_max_b]*nbr_elements[2] + [v_max_w]*nbr_elements[3] + [255]*nbr_elements[4] + \
            [100]*nbr_elements[5] + [100]*nbr_elements[6] + [100]*nbr_elements[7] + [azimuth_max]*nbr_elements[8] + [100]*nbr_elements[9] + [100]*nbr_elements[10]
   
            #NOTE: There are multiple places to change the lower/upper bounds if they are decided to be changed! Here and in simulation_methods. Be careful!
           
            creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0)) #Assuming two objectives.
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

            toolbox = base.Toolbox()

            toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP) #The registering is done above main in the example.
            toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", evaluate())
            toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0) #Lower eta to make the child more different from the parent.
            toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/len(BOUND_LOW))
            toolbox.register("select", tools.selNSGA2)

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean, axis=0)
            stats.register("std", numpy.std, axis=0)
            stats.register("min", numpy.min, axis=0)
            stats.register("max", numpy.max, axis=0)

            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"

            pop = toolbox.population(n=pop_size)

            #Need to run_simulation once for each individual in pop. Set weather in between.
            for ind in pop:
                (starting_frames_ts,speed_dict,color_dict,weather_dict) = generate_values(bool_dict, nbr_car_tracks, nbr_bike_tracks, nbr_walker_tracks, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames_ts, weather_dict, speed_dict, track_dict, color_dict, starting_frames_ts, max_frames, ind)
                starting_frames = np.rint(starting_frames_ts*delta_timestamps/delta_seconds) #At what frames to spawn actors.

                weather = set_weather_params(weather_dict)
               
                world.set_weather(weather)

                run_simulation(gen_nbr, total_nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library)
           
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # This is just to assign the crowding distance to the individuals
            # no actual selection is done
            pop = toolbox.select(pop, len(pop))

            record = stats.compile(pop)
            logbook.record(gen=0, evals=len(invalid_ind), **record)
            print(logbook.stream)

        else:
            (starting_frames_ts,speed_dict,color_dict,weather_dict) = generate_values(bool_dict, nbr_car_tracks, nbr_bike_tracks, nbr_walker_tracks, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames_ts, weather_dict, speed_dict, track_dict, color_dict, starting_frames_ts, max_frames, pop)
            starting_frames = np.rint(starting_frames_ts*delta_timestamps/delta_seconds) #At what frames to spawn actors.

            #Save input parameters.
            param_save = f"input_params_temp/batch_4.txt"
            os.makedirs(os.path.dirname(param_save), exist_ok=False)
            with open(param_save, 'a') as f:
                str1 = f"{gen_nbr}:   "
                for car in color_dict:
                    str1 += 'r: ' + str(color_dict[car][0]) + "   "
                    str1 += 'g: ' + str(color_dict[car][1]) + "   "
                    str1 += 'b: ' + str(color_dict[car][2]) + "   "
                    

                str1 += 'az: ' + str(np.round(weather_dict['sun_azimuth_angle'],2)) + "   "
                str1 += 'cl: ' + str(np.round(weather_dict['cloudiness'],2)) + "\n"

                f.writelines([str1])
            #Save input parameters.
            
            weather = set_weather_params(weather_dict)
           
            world.set_weather(weather)

            run_simulation(gen_nbr, total_nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library)

       #Save as video
        image_folder = 'images'

        for i in range(2):
            images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith(f"{i}")]
            if i == 0:
                video_name = f'output_scenario/scenario_{gen_nbr}_left.mp4' 
            else:
                video_name = f'output_scenario/scenario_{gen_nbr}_right.mp4'
            frame = cv2.imread(os.path.join(image_folder, images[0]))

            try:
                height, width, layers = frame.shape

            except AttributeError:
                print(os.getcwd())
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video = cv2.VideoWriter(video_name, fourcc, 20, (width,height))
            count = 0

            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))
                print(count)
                count += 1

            cv2.destroyAllWindows()
            video.release()

        shutil.rmtree(image_folder)
        #Save as video
    

        print('Starting the loop...')
       
        for gen_nbr in range(1,nbr_generations):
            if model == 'nsga2':
                # Vary the population
                offspring = tools.selTournamentDCD(pop, len(pop))
                offspring = [toolbox.clone(ind) for ind in offspring] #Need this since otherwise we make changes to the individuals in pop.

                for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() <= crosspver_prob:
                        toolbox.mate(ind1, ind2)

                    toolbox.mutate(ind1)
                    toolbox.mutate(ind2)
                    del ind1.fitness.values, ind2.fitness.values

                for ind in offspring:
                    (starting_frames_ts,speed_dict,color_dict,weather_dict) = generate_values(bool_dict, nbr_car_tracks, nbr_bike_tracks, nbr_walker_tracks, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames_ts, weather_dict, speed_dict, track_dict, color_dict, starting_frames_ts, max_frames, ind)
                    starting_frames = np.rint(starting_frames_ts*delta_timestamps/delta_seconds) #At what frames to spawn actors.
                       
                    weather = set_weather_params(weather_dict)
                    world.set_weather(weather)

                    run_simulation(gen_nbr, total_nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
           
                # Select the next generation population
                pop = toolbox.select(pop + offspring, pop_size)
                record = stats.compile(pop)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                print(logbook.stream)

            else:
                (starting_frames_ts,speed_dict,color_dict,weather_dict) = generate_values(bool_dict, nbr_car_tracks, nbr_bike_tracks, nbr_walker_tracks, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames_ts, weather_dict, speed_dict, track_dict, color_dict, starting_frames_ts, max_frames, pop)
                starting_frames = np.rint(starting_frames_ts*delta_timestamps/delta_seconds) #At what frames to spawn actors.

                with open(param_save, 'a') as f:
                    str1 = f"{gen_nbr}:   "
                    for car in color_dict:
                        str1 += 'r: ' + str(color_dict[car][0]) + "   "
                        str1 += 'g: ' + str(color_dict[car][1]) + "   "
                        str1 += 'b: ' + str(color_dict[car][2]) + "   "

                    str1 += 'az: ' + str(np.round(weather_dict['sun_azimuth_angle'],2)) + "   "
                    str1 += 'cl: ' + str(np.round(weather_dict['cloudiness'],2)) + "\n"

                    f.writelines([str1])
                weather = set_weather_params(weather_dict)
                world.set_weather(weather)

                run_simulation(gen_nbr, total_nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library)

            #Save as video
            image_folder = 'images'
            for i in range(2):
                images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith(f"{i}")]
                if i == 0:
                    video_name = f'output_scenario/scenario_{gen_nbr}left.mp4'
                else:
                    video_name = f'output_scenario/scenario_{gen_nbr}right.mp4'
                frame = cv2.imread(os.path.join(image_folder, images[0]))

                try:
                    height, width, layers = frame.shape
                except AttributeError:
                    print(os.getcwd())
                    
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                video = cv2.VideoWriter(video_name, fourcc, 20, (width,height))
                count = 0

                for image in images:
                    video.write(cv2.imread(os.path.join(image_folder, image)))
                    print(count)
                    count += 1
                
                cv2.destroyAllWindows()
                video.release()

            shutil.rmtree(image_folder)
            #Save as video
                     
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
