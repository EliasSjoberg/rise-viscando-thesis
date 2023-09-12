import numpy as np
import carla
import random
import time
from queue import Queue
from queue import Empty
import os
import cv2
import shutil
import csv

import pandas as pd

def save_params(batch_idx, ind_nbr, color_dict, weather_dict):
    """
    Saves ground truth parameter values in a text file. The method only saves RGB-values, the azimuth angle and the cloudiness parameter.

    Args:
        batch_idx (int): Determines which of the vehicle_tracks to simulate. Ranges from 0 to 3.
        ind_nbr (int): Index of the individual. Ranges from 0 to pop_size-1.
        color_dict (dict): Keeps track of the colors of the cars in the simulation.
        weather_dict (dict: Keeps track of the weather parameters in the simulation.
    """
    param_save = f"input_params/batch_{batch_idx+1}.txt"
    os.makedirs(os.path.dirname(param_save), exist_ok=True)
    with open(param_save, 'a') as f:
        str1 = f"{ind_nbr}:   "
        for car in color_dict:
            str1 += 'r: ' + str(round(color_dict[car][0])) + "   "
            str1 += 'g: ' + str(round(color_dict[car][1])) + "   "
            str1 += 'b: ' + str(round(color_dict[car][2])) + "   "

        str1 += 'az: ' + str(np.round(weather_dict['sun_azimuth_angle'],2)) + "   "
        str1 += 'cl: ' + str(np.round(weather_dict['cloudiness'],2)) + "\n"

        f.writelines([str1])

def save_video(batch_idx, ind_nbr):
    """
    Creates a video out of images saved in a folder
    
     Args:
        batch_idx (int): Determines which of the vehicle_tracks to simulate. Ranges from 0 to 3.
        ind_nbr (int): Index of the individual. Ranges from 0 to pop_size-1.
    """
    image_folder = 'images'
    for i in range(2):
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith(f"{i}")]
        if i == 0:
            video_name = f'output_scenario_{batch_idx}/scenario_{ind_nbr}left.mp4'
        else:
            video_name = f'output_scenario_{batch_idx}/scenario_{ind_nbr}right.mp4'
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


def compute_errors(batch, nbr_actors, pop_size, current_gen, get_positions, path_op, path_gt):
    """
    Computes the distance, speed and classification errors for each individual/scenario in the population.
    Returns error_values_dist, error_values_speed and error_values_perc, which contain the respective error
    for each trajectory (all of sizes 20X7), and avg_errors, which averages the errors across the trajectories (of size 20X3).
    avg_error contains the distance error, speed error and classification error in that order.
    If get_positions==True the positions and the errors in the x/y-directions are returned. This is only done in the evaluation, not when running scenarios.

    Args:
        batch (int): Determines which of the vehicle_tracks/model combination to simulate. Ranges from 0 to 3.
        nbr_actors (int): The number of actors (cars, bikes, walkers) participating in the simulation.
        pop_size (int): Population size.
        current_gen (int): The current generation.
        get_positions (bool): Whether to return the positions and the errors in the x/y-directions.
        path_op (str): Path to OTUS3D estimations.
        path_gt (str): Path to ground truth values.
    """

    path_est = path_op + f'/output_scenario_gen{current_gen-1}_0-3_tracks_csv' #Path to OTUS3D estimations
    path_gt_batch = path_gt + f'/ground_truth_gen{current_gen-1}_{batch}' #Path to ground truth positions, speeds, classifications and timestamp values

    avg_errors = []

    error_values_dist = []
    error_values_speed = []
    error_values_perc = []
    step_perc = []
    ts_save_all = []

    coords_gt_save = []
    coord_diff = []
    speed_diff = []
    class_diff = []

    for ind_idx in range(pop_size):
        fileop = open(path_est + f'/scenario_{batch}_{ind_idx}.csv')
        filegt = open(path_gt_batch + f'/ground_truth_{ind_idx}.csv')
        csvreaderop = csv.reader(fileop, delimiter=';')
        csvreadergt = csv.reader(filegt, delimiter=';')
        headerop = next(csvreaderop)
        headergt = next(csvreadergt)

        row_op = next(csvreaderop) #The first row in the output/estimation
        op_dict = dict()

        while True:
            #Read the output and put the values in a dictionary op_dict
            id_op = row_op[0]
            if op_dict.get(id_op) is None:
                op_dict[id_op] = dict()
                op_dict[id_op]['ts'] = []
                op_dict[id_op]['x'] = []
                op_dict[id_op]['y'] = []
                op_dict[id_op]['speed'] = []
                op_dict[id_op]['type'] = []

            op_dict[id_op]['ts'].append(float(row_op[1][17:23]))
            op_dict[id_op]['x'].append(float(row_op[2]) + 5)
            op_dict[id_op]['y'].append(float(row_op[3]))
            op_dict[id_op]['speed'].append(np.sqrt(float(row_op[4])**2 + float(row_op[5])**2))
            op_dict[id_op]['type'].append(int(row_op[6]))
            
            if len(op_dict[id_op]['ts']) > 1 and op_dict[id_op]['ts'][-1] != round(op_dict[id_op]['ts'][-2] + 0.05,2):
                print('jump in timestamp!')
                break

            if op_dict[id_op]['type'][-1] == 18:
                op_dict[id_op]['type'][-1] = 2

            elif op_dict[id_op]['type'][-1] == 4:
                op_dict[id_op]['type'][-1] = 0            
            
            try:
                row_op = next(csvreaderop)


            except StopIteration:
                break

        row_gt = next(csvreadergt)

        result_dict_dist = dict()
        result_dict_speeds = dict()
        result_dict_types = dict()
        result_dict_key = dict()
        result_dict_steps = dict()

        #result_dict_types_save = dict()
        ts_save = dict()

        result_dict_diff = dict() #checking diff

        """
        The ground truth and estimated trajectories are linked together according to the following method:
        First, common timestamps are found. For each gt/op trajectory pair that has a timestamp in common, the errors in
        distance, speed and classification are computed. These errors are all put in dictionaries. The trajectory pair
        that has the smallest distance error is chosen as the pair to link together. It is the errors between these two
        trajectories that are chosen as contributing to the fitness value.
        """

        stop = False

        while stop == False: #First, we loop over the ground truth data
            id_gt = row_gt[0]
            id_check = id_gt
            timestamps_gt = np.zeros(500)
            x_gt = np.zeros(500)
            y_gt = np.zeros(500)
            speeds_gt = np.zeros(500)
            types_gt = np.zeros(500)
            idx = 0
            
            while id_check == id_gt:
                #Fetch ground truth data for actor/trajectory with index id_gt
                timestamps_gt[idx] = row_gt[1]
                
                x_gt[idx] = float(row_gt[2])
                y_gt[idx] = row_gt[3]
                
                speeds_gt[idx] = row_gt[4]
                types_gt[idx] = row_gt[5]
                idx += 1
                try:
                    row_gt = next(csvreadergt)
                    id_check = row_gt[0]
                except StopIteration:
                    stop = True #Stop the loop when there are no more actors in gt
                    break

            nbr_points = idx

            ts_save[id_gt] = timestamps_gt[:nbr_points]

            for ids in op_dict: #Loop over all estimated trajectories to compare with gt trajectory with index id_gt
                for idx_gt in range(nbr_points):
                    if timestamps_gt[idx_gt] in op_dict[ids]['ts']:
                        idx_op = op_dict[ids]['ts'].index(timestamps_gt[idx_gt])
                        nbr_steps = min(nbr_points-idx_gt, len(op_dict[ids]['ts'])-idx_op)
                        
                        op_x = np.array(op_dict[ids]['x'][idx_op:idx_op+nbr_steps])
                        op_y = np.array(op_dict[ids]['y'][idx_op:idx_op+nbr_steps])
                        op_speeds = np.array(op_dict[ids]['speed'][idx_op:idx_op+nbr_steps])
                        op_types = np.array(op_dict[ids]['type'][idx_op:idx_op+nbr_steps])

                        error_dist = np.mean(np.sqrt((op_x-x_gt[idx_gt:idx_gt+nbr_steps])**2 + (op_y-y_gt[idx_gt:idx_gt+nbr_steps])**2))
                        error_speed = np.mean(np.abs(op_speeds-speeds_gt[idx_gt:idx_gt+nbr_steps]))
                        wrong_class = np.sum(op_types != types_gt[idx_gt:idx_gt+nbr_steps])/len(op_types)


                        if id_gt not in result_dict_dist:
                            result_dict_dist[id_gt] = [error_dist] 
                            result_dict_speeds[id_gt] = [error_speed]
                            result_dict_types[id_gt] = [wrong_class]
                            result_dict_key[id_gt] = [ids]
                            result_dict_steps[id_gt] = [nbr_steps/nbr_points]
                            #result_dict_types_save[id_gt] = [np.stack((op_types,types_gt[idx_gt:idx_gt+nbr_steps]))]

                            if get_positions == True:
                                diff = np.array((op_x-x_gt[idx_gt:idx_gt+nbr_steps],op_y-y_gt[idx_gt:idx_gt+nbr_steps])) #two rows, nbr_steps columns
                                result_dict_diff[id_gt] = dict()
                                result_dict_diff[id_gt]['diff'] = [diff]
                                result_dict_diff[id_gt]['coords'] = [np.stack((x_gt[idx_gt:idx_gt+nbr_steps],y_gt[idx_gt:idx_gt+nbr_steps]))]
                                result_dict_diff[id_gt]['speeds'] = [np.abs(op_speeds-speeds_gt[idx_gt:idx_gt+nbr_steps])]
                                result_dict_diff[id_gt]['types'] = [op_types != types_gt[idx_gt:idx_gt+nbr_steps]]
                                #result_dict_diff[id_gt]['steps'] = [nbr_steps/nbr_points]
                                                            
                        else:
                            result_dict_dist[id_gt].append(error_dist)
                            result_dict_speeds[id_gt].append(error_speed)
                            result_dict_types[id_gt].append(wrong_class)
                            result_dict_key[id_gt].append(ids)
                            result_dict_steps[id_gt].append(nbr_steps/nbr_points)
                            
                            #result_dict_types_save[id_gt].append(np.stack((op_types,types_gt[idx_gt:idx_gt+nbr_steps])))

                            if get_positions == True:
                                diff = np.array((op_x-x_gt[idx_gt:idx_gt+nbr_steps],op_y-y_gt[idx_gt:idx_gt+nbr_steps])) #two rows, nbr_steps columns
                                result_dict_diff[id_gt]['diff'].append(diff)
                                result_dict_diff[id_gt]['coords'].append(np.stack((x_gt[idx_gt:idx_gt+nbr_steps],y_gt[idx_gt:idx_gt+nbr_steps])))
                                result_dict_diff[id_gt]['speeds'].append(np.abs(op_speeds-speeds_gt[idx_gt:idx_gt+nbr_steps]))
                                result_dict_diff[id_gt]['types'].append(op_types != types_gt[idx_gt:idx_gt+nbr_steps])
                                #result_dict_diff[id_gt]['steps'].append(nbr_steps/nbr_points)
                                #two rows, nbr_steps columns

                        break

        min_dists = np.zeros(len(result_dict_dist))
        min_speeds = np.zeros(len(result_dict_dist))
        perc_misclass = np.zeros(len(result_dict_dist))
        corresponding_id = np.zeros(len(result_dict_dist))
        step_perc_ind = np.zeros(len(result_dict_dist))

        coord_diff.append([])
        speed_diff.append([])
        class_diff.append([])
        coords_gt_save.append([])
        
        for id_gt in result_dict_dist:        
            min_dist = min(result_dict_dist[id_gt])
            index_min = min(range(len(result_dict_dist[id_gt])), key=result_dict_dist[id_gt].__getitem__)
            min_dists[int(id_gt)] = min_dist
            min_speeds[int(id_gt)] = result_dict_speeds[id_gt][index_min]
            perc_misclass[int(id_gt)] = result_dict_types[id_gt][index_min]
            corresponding_id[int(id_gt)] = result_dict_key[id_gt][index_min]
            step_perc_ind[int(id_gt)] = result_dict_steps[id_gt][index_min]

            if get_positions == True:
                coord_diff[ind_idx].append(result_dict_diff[id_gt]['diff'][index_min])
                speed_diff[ind_idx].append(result_dict_diff[id_gt]['speeds'][index_min])
                class_diff[ind_idx].append(result_dict_diff[id_gt]['types'][index_min])
                coords_gt_save[ind_idx].append(result_dict_diff[id_gt]['coords'][index_min])

            
        avg_errors.append((np.mean(min_dists), np.mean(min_speeds), np.mean(perc_misclass)))


        error_values_dist.append(min_dists)
        error_values_speed.append(min_speeds)
        error_values_perc.append(perc_misclass)
        step_perc.append(step_perc_ind)
        temp = [s[0] for s in list(ts_save.values())]
        ts_save_all.append(temp)
        
        
    

    if get_positions == True:
        return (avg_errors, error_values_dist, error_values_speed, error_values_perc, step_perc, coord_diff, speed_diff, class_diff, coords_gt_save)
    else:
        return (avg_errors, error_values_dist, error_values_speed, error_values_perc, ts_save_all)#step_perc)


def generate_values(bool_dict, nbr_cars, nbr_bikes, nbr_walkers, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames, weather_dict, speed_dict, track_dict, color_dict, starting_frames, max_frames, ind):
    """
    "Generates" starting time, speed, color and weather values used in the simulation and puts them in starting_frames, speed_dict, color_dict and
    weather_dict respectively. Initially, these dictionaries contain default values.
    The method only generates values for variable parameters, as decided by bool_dict.
    If len(ind)==0 (which is the case when the baseline model is used) the method generates random values, otherwise the values are fetched from ind.
    Returns starting_frames, speed_dict, color_dict, weather_dict

    Args:
       bool_dict (dict): Determines which parameters should be variable across generations. True means that the parameter is variable
       nbr_cars (int): Number of cars participating in the simulation.
       nbr_bikes (int): Number of bikes participating in the simulation.
       nbr_walkers (int): Number of walkers participating in the simulation.
       nbr_vel_points (int): Number of points in which to vary velocities (only when velocities are set as variable).
       v_max (int): max speed allowed for cars (m/s) (only when velocities are set as variable).
       v_max_b (int): max speed allowed for bikes (m/s) (only when velocities are set as variable).
       v_max_w (int): max speed allowed for walkers (m/s) (only when velocities are set as variable).
       nbr_frames (int): Number of "long frames" putting an upper bound of when the creation of an actor is allowed.
       weather_dict (dict): Keeps track of the weather parameters in the simulation.
       speed_dict (dict): Keeps track of the speed values for each actor to target at waypoints in the simulation.
       track_dict (dict): Keeps track of the indices of the trajectories.
       color_dict (dict): Keeps track of the colors of the cars in the simulation.
       starting_frames (np.array): The frames at which each actor in the simulation should spawn (in "long frames").
       max_frames (np.array): The number of waypoints or "long frames" in each trajectory
       ind (list): Contains the parameter values for an individual in the population (i.e. a scenario).
    """
    nbr_actors = nbr_cars + nbr_bikes + nbr_walkers

    frame_durations = np.zeros(nbr_actors, dtype = np.uint32)
    ind_idx = 0 #Keeping track of the index of ind

    if bool_dict['timestamps'] == True:
        if len(ind) == 0: #We are using the baseline model.
            starting_frames = np.random.randint(0,nbr_frames+1,nbr_actors)
        else: #We are using the nsga model
            starting_frames[:nbr_actors] = ind[0:nbr_actors]
            ind_idx = nbr_actors

           
    if bool_dict['vels'] == True:
        for idx in range(nbr_actors):
            frame_durations[idx] = max_frames[track_dict[idx]]
        
        if len(ind) == 0:
            #If the minimum speeds are changed (from 0.5,0.5,0.3), they should also be changed in BOUND_LOW in simulation_main.
            temp1 = 0.5 + (v_max-0.5)*np.random.rand(nbr_cars,nbr_vel_points) #Each row corresponds to one track
            temp2 = 0.5 + (v_max_b-0.5)*np.random.rand(nbr_cars,nbr_vel_points)
            temp3 = 0.3 + (v_max_w-0.3)*np.random.rand(nbr_walkers,nbr_vel_points)
            temp = np.vstack((temp1,temp2,temp3))
            speed_dict = dict()
        else:
            temp = np.array(list(ind[ind_idx:ind_idx+nbr_actors*nbr_vel_points]))
            temp = np.reshape(temp,(nbr_actors,nbr_vel_points))
            ind_idx += nbr_actors*nbr_vel_points
           
        for i in range(nbr_actors):
            speed_dict[i] = np.zeros(frame_durations[i])
            delta = int(np.floor(len(speed_dict[i])/(nbr_vel_points-1)))
            for j in range(nbr_vel_points-1):
                speed_dict[i][j*delta:(j+1)*delta] = np.linspace(temp[i][j],temp[i][j+1],delta)
            speed_dict[i][j*delta:] = speed_dict[i][j*delta-1]

    idx = 0       
   
    if bool_dict['colors'] == True:
        #Note that bp_list should only include cars.
        if len(ind) == 0:
            temp = np.random.randint(0,256,(len(color_dict),3)) #Each row corresponds to one car. RGB.
        else:
            temp = np.array(list(ind[ind_idx:ind_idx+len(color_dict)*3]))
            temp = np.reshape(temp,(len(color_dict),3))
            ind_idx += len(color_dict)*3

        for idx in color_dict:
            color_dict[idx] = list(temp[idx][:])
       
       
    bool_array = [bool_dict['sun_angle'],bool_dict['clouds'],bool_dict['precip'],bool_dict['wind'],bool_dict['fog'],bool_dict['wetness']]
    azimuth_min = 25 #If changed, also needs to be changed in simulation_main.
    azimuth_max = 265 - (azimuth_min - 15)
    min_limits = [azimuth_min,0,0,0,0,0]
    max_limits = [azimuth_max,100,100,100,100,100]
    #If any of these are changed, they also need to be changed in BOUND_LOW/BOUND_UP simulation_main
    weathers = ['sun_azimuth_angle','cloudiness','precipitation','wind_intensity','fog_density','wetness']
   
    idx = 0

    for b in bool_array:
        if b == True:
            if len(ind) == 0:
                temp = min_limits[idx] + (max_limits[idx]-min_limits[idx])*np.random.rand()
                weather_dict[weathers[idx]] = temp
            else:
                temp = ind[ind_idx:ind_idx+1]
                ind_idx += 1
                weather_dict[weathers[idx]] = temp[0]
        idx += 1

    return (starting_frames,speed_dict,color_dict,weather_dict)
       
   

def set_weather_params(weather_dict: dict):
    """
    Sets the weather parameters in the simulation according to weather_dict.

    Args:
        weather_dict (dict): Keeps track of the weather parameters in the simulation.
    """
    weather = carla.WeatherParameters(
        cloudiness = weather_dict['cloudiness'],
        precipitation = weather_dict['precipitation'],
        wind_intensity = weather_dict['wind_intensity'],
        sun_azimuth_angle = weather_dict['sun_azimuth_angle'],
        sun_altitude_angle = -50/15625*(weather_dict['sun_azimuth_angle']-15)*(weather_dict['sun_azimuth_angle']-265), #140 corresponds to midday.
        wetness = weather_dict['wetness'], #Wetness of road. Don't think it affects the cars.
        fog_density = weather_dict['fog_density'],
        fog_distance = weather_dict['fog_distance'],
        fog_falloff = weather_dict['fog_falloff'],
        scattering_intensity = weather_dict['scattering_intensity'] #Makes the light more affected by the fog
        )
    return weather


def sensor_callback(sensor_data, sensor_queue, sensor_nbr):
    """
    Method to save the captured images.

    Args:
        sensor_data (carla.Image): The image as raw data.
        sensor_queue (Queue): A synchronized FIFO (first-in, first-out) queue, so that all images are saved in the order they were captured.
        sensor_nbr (int): To keep track of the two cameras. 0 for the left camera and 1 for the right.
    """
    sensor_data.save_to_disk('images/%d-%.6d.jpg' % (sensor_nbr, sensor_data.frame))
    print(f"Put in queue at {sensor_data.frame}")
    sensor_queue.put((sensor_data.frame, sensor_nbr))

def run_simulation(ind_nbr, batch_idx, nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library):
    """
    Runs one traffic scenario, captures and saves images and saves the ground truth data.

    Args:
        ind_nbr (int): Index of the individual. Ranges from 0 to pop_size-1.
        batch_idx (int): Determines which of the vehicle_tracks to simulate. Ranges from 0 to 3.
        nbr_tracks (int): The number of trajectories or actors (cars, bikes, walkers) participating in the simulation.
        coord_dict (dict): Keeps track of the xy-coordinates for each actor to target at waypoints in the simulation.
        speed_dict (dict): Keeps track of the speed values for each actor to target at waypoints in the simulation.
        delta_seconds (float): "Short frames". Amount of simulated time per frame. Must be 1, 1/2, 1/3 etc of delta_timestamps.
        delta_timestamps (float): "Long frames". The amount of time between capturing of images and also the time between each waypoint in the trajectories.
        starting_frames (np.array): The frames at which each actor in the simulation should spawn (in "short frames").
        object_classes (np.array): Keeping track of whether the actor is a car, bicycle or walker.
        color_dict (dict): Keeps track of the colors of the cars in the simulation.
        world (carla.world): The world client.
        sensor_list (list): List containing the cameras.
        sensor_queue (Queue): A synchronized FIFO (first-in, first-out) queue, so that all images are saved in the order they were captured.
        blueprint_library (carla.BlueprintLibrary): Contains blueprint models for actors.
    """
    stop = False
    is_active = np.zeros(nbr_tracks) #0 means not active yet, 1 means active and 2 means not active anymore.
    frame_inds = np.zeros(nbr_tracks)
    actor_dict = dict()
    spawn_waiting = 20 #How many frames to wait between spawning and driving.
    frame_idx_glob = 0
    
    output_inds = np.zeros(nbr_tracks, dtype = np.uint32) #Keeping track of the number of waypoints where ground truth has been collected
    coords_gt = dict() #To save ground truth
    speeds_gt = dict()
    timestamps_gt = dict()
    framenbrs_gt = dict()

    start_frame = world.get_snapshot().frame
    
    transx = 11 #To change between Viscando coordinates and CARLA coordinates.
    transy = -9

       
    while stop == False:
        for track_idx in range(nbr_tracks): #Loop over each actor
            if is_active[track_idx] == 0 and frame_idx_glob + spawn_waiting >= starting_frames[track_idx]:
                #Spawn actor               
                coords = coord_dict[track_idx]
                
                if object_classes[track_idx] == 18: #Corresponds to car
                    color = color_dict[track_idx]
                    ego_bp = blueprint_library.find('vehicle.tesla.model3')
                    ego_bp.set_attribute('color',f"{round(color[0])},{round(color[1])},{round(color[2])}")
                    start_trans = carla.Transform()
                    start_trans.location = carla.Location(coords[0,0],coords[1,0], 0.5)
                    
                elif object_classes[track_idx] == 1: #Corresponds to bike
                    ego_bp = blueprint_library.find('vehicle.bh.crossbike')
                    start_trans = carla.Transform()
                    start_trans.location = carla.Location(coords[0,0],coords[1,0], 0.5)
                    
                else: #Corresponds to walker
                    ego_bp = blueprint_library.find('walker.pedestrian.0001')
                    start_trans = carla.Transform()
                    start_trans.location = carla.Location(coords[0,0],coords[1,0], 1)
                   
                start_trans.rotation.yaw = np.degrees(np.arctan((coords[1,1]-coords[1,0])/(coords[0,1]-coords[0,0])))
                
                #Note that in Carla clockwise rotation is positive when viewed from above.
                if coords[0,1] < coords[0,0]:        
                    start_trans.rotation.yaw += 180
                ego_actor = world.try_spawn_actor(ego_bp, start_trans)

                if ego_actor is None:
                    print('spawn collision')
                    starting_frames[track_idx] += 30
                    continue #Collision at spawn. Try again next frame.
               
                actor_dict[track_idx] = ego_actor
                is_active[track_idx] = 1
                coords_gt[track_idx] = np.zeros((2,400))
                speeds_gt[track_idx] = np.zeros(400)
                timestamps_gt[track_idx] = np.zeros(400)
                framenbrs_gt[track_idx] = np.zeros(400)
                

            elif is_active[track_idx] == 1 and frame_idx_glob >= starting_frames[track_idx]:
                #Move actor forward
                coords = coord_dict[track_idx]
                ego_actor = actor_dict[track_idx]
                idx = int(np.ceil((frame_inds[track_idx] + 1)/round(delta_timestamps/delta_seconds)))
                current_loc = ego_actor.get_location()
                current_fv = ego_actor.get_transform().get_forward_vector()
                
                if frame_idx_glob % np.rint(delta_timestamps/delta_seconds) == 0: #Image is captured this frame; thus ground truth should be collected
                    try:
                        output_idx = output_inds[track_idx]
                        coords_gt[track_idx][0,output_idx] = current_loc.x - transx #Ground truth is saved in Viscando's coordinates
                        coords_gt[track_idx][1,output_idx] = current_loc.y - transy
                        speeds_gt[track_idx][output_idx] = ego_actor.get_velocity().length()
                        timestamps_gt[track_idx][output_idx] = frame_idx_glob*delta_seconds
                        framenbrs_gt[track_idx][output_idx] = world.get_snapshot().frame
                    except IndexError:
                        size = len(speeds_gt[track_idx])
                        new_coords = np.zeros((2,size+100))
                        new_speeds = np.zeros(size+100)
                        new_timestamps = np.zeros(size+100)
                        new_framenbrs = np.zeros(size+100)
                        
                        new_coords[:,:output_idx] = coords_gt[track_idx][:,:output_idx]
                        new_coords[0,output_idx] = current_loc.x - transx
                        new_coords[1,output_idx] = current_loc.y - transy
                        new_speeds[:output_idx] = speeds_gt[track_idx][:output_idx]
                        new_speeds[output_idx] = ego_actor.get_velocity().length()
                        new_timestamps[:output_idx] = timestamps_gt[track_idx][:output_idx]
                        new_timestamps[output_idx] = frame_idx_glob*delta_seconds,2
                        new_framenbrs[:output_idx] = framenbrs_gt[track_idx][:output_idx]
                        new_framenbrs[output_idx] = world.get_snapshot().frame
                         
                        coords_gt[track_idx] = new_coords
                        speeds_gt[track_idx] = new_speeds
                        timestamps_gt[track_idx] = new_timestamps
                        framenbrs_gt = new_framenbrs

                    output_inds[track_idx] += 1
                    print(world.get_snapshot().frame)


                next_fv = carla.Vector3D(coords[0,idx]-current_loc.x,coords[1,idx]-current_loc.y,0)
                next_fv = next_fv/next_fv.length()

                if object_classes[track_idx] == 2: #Walker
                    control = carla.WalkerControl()
                    control.speed = speed_dict[track_idx][idx]
                    control.direction = next_fv
                    ego_actor.apply_control(control)

                else: #Car or bike
                    #Set velocity
                    ego_actor.set_target_velocity(next_fv*speed_dict[track_idx][idx])
                    #Set steering
                    contr = carla.VehicleControl()
                    current_fv.z = 0
                    current_fv = current_fv/current_fv.length()
                    angle = np.arccos(current_fv.dot(next_fv)) #Angle between the current and future velocities.
                   
                    angle_current_fv = -np.arctan(current_fv.y/current_fv.x)
                    if current_fv.x < 0:
                        angle_current_fv += np.pi
                    angle_next_fv = -np.arctan(next_fv.y/next_fv.x)
                    if next_fv.x < 0:
                        angle_next_fv += np.pi

                    if angle_current_fv > angle_next_fv:
                        if angle > np.pi/2:
                            contr.steer = 1
                        else:
                            contr.steer = 2*angle/np.pi
                    else:
                        if angle > np.pi/2:
                            contr.steer = -1
                        else:
                            contr.steer = -2*angle/np.pi
                   
                    ego_actor.apply_control(contr)
                   
                frame_inds[track_idx] += 1

                if frame_inds[track_idx] + 1 >= round(delta_timestamps*len(speed_dict[track_idx])/delta_seconds) - (round(delta_timestamps/delta_seconds) - 1):
                    is_active[track_idx] = 2 #Done
                    ego_actor.destroy()

        if np.all(is_active == 2):
            stop = True
            end_frame = start_frame + frame_idx_glob
            while (end_frame - start_frame) % np.rint(delta_timestamps/delta_seconds) != 0: #To keep the image capturing in sync for the next scenario
                world.tick()
                end_frame += 1

        else: #Write frame numbers to a text document according to Viscando's wishes.
            filename = f"output_scenario_{batch_idx}/scenario_{ind_nbr}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'a') as f:
                str1 = str(frame_idx_glob)
                str1 += "      "

                secs = int(np.floor(frame_idx_glob*delta_timestamps))
                str2 = str(secs)
                str2 += "      "

                diff = (frame_idx_glob*delta_timestamps - secs)/delta_timestamps
                
                microsecs = int(50000*np.round((frame_idx_glob*delta_timestamps - secs)/delta_timestamps))

                str3 = str(microsecs)
                str3 += "\n"

                f.writelines([str1, str2, str3])
                
            world.tick()

            
            if frame_idx_glob % np.rint(delta_timestamps/delta_seconds) == 0: #Capture image this frame
                try:
                    for _ in range(len(sensor_list)):
                        s_frame = sensor_queue.get(True,10.0) #Could increase this waiting time.
                except Empty:
                    print(f"Some data was lost, {world.get_snapshot().frame}")
            
            frame_idx_glob += 1
    print(frame_idx_glob)

    

    
    ###Saving the ground truth to a csv file.
    gt_output = np.zeros((np.sum(output_inds),nbr_tracks)) 
    idx_save = 0
    for track_idx in range(nbr_tracks):
        idx = output_inds[track_idx]

        gt_output[idx_save:idx_save+idx,0] = track_idx
        gt_output[idx_save:idx_save+idx,1] = timestamps_gt[track_idx][:idx]
        gt_output[idx_save:idx_save+idx,2] = coords_gt[track_idx][0,:idx]
        gt_output[idx_save:idx_save+idx,3] = coords_gt[track_idx][1,:idx]
        gt_output[idx_save:idx_save+idx,4] = speeds_gt[track_idx][:idx]

        if object_classes[track_idx] == 2:
            object_class = 0

        elif object_classes[track_idx] == 18:
            object_class = 2
        else:
            object_class = 1
            
        gt_output[idx_save:idx_save+idx,5] = object_class
        
        gt_output[idx_save:idx_save+idx,6] = framenbrs_gt[track_idx][:idx]

        idx_save += idx

    df = pd.DataFrame(gt_output, columns=['ID', 'Time', 'X', 'Y', 'Speed', 'Type', 'Frame'])
    df['ID'] = df['ID'].apply(np.uint32)
    df['Type'] = df['Type'].apply(np.uint32)
    df['Frame'] = df['Frame'].apply(np.uint32)

    filename = f"ground_truth_{batch_idx}/ground_truth_{ind_nbr}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, sep = ';', index = None)

