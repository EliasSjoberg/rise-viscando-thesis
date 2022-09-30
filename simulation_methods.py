import numpy as np
import carla
import random
import time
from queue import Queue
from queue import Empty
import os

import pandas as pd


def generate_values(bool_dict, nbr_cars, nbr_bikes, nbr_walkers, nbr_vel_points, v_max, v_max_b, v_max_w, nbr_frames, weather_dict, speed_dict, track_dict, color_dict, starting_frames, max_frames, pop):

    """
    Puts simulation values into dictionaries. If pop is empty, random values are generated. Otherwise, it is assumed that pop contains
    The parameter values that should be used.
    returns: starting_frames, speed_dict, color_dict, weather_dict
    """
    
    nbr_actors = nbr_cars + nbr_bikes + nbr_walkers
    azimuth_min = 25
    azimuth_max = 265 - (azimuth_min - 15)

    frame_durations = np.zeros(nbr_actors, dtype = np.uint32)
    pop_idx = 0

    if bool_dict['timestamps'] == True:
        if len(pop) == 0: #We are using the base model.
            starting_frames = np.random.randint(0,nbr_frames+1,nbr_actors)
        else:
            starting_frames[:nbr_actors] = pop[0:nbr_actors]
            pop_idx = nbr_actors

           
    if bool_dict['vels'] == True:
        for idx in range(nbr_actors):
            frame_durations[idx] = max_frames[track_dict[idx]]
        
        if len(pop) == 0:
            temp1 = 0.5 + (v_max-0.5)*np.random.rand(nbr_cars,nbr_vel_points) #Each row corresponds to one track
            temp2 = 0.5 + (v_max_b-0.5)*np.random.rand(nbr_cars,nbr_vel_points)
            temp3 = 0.3 + (v_max_w-0.3)*np.random.rand(nbr_walkers,nbr_vel_points)
            temp = np.vstack((temp1,temp2,temp3))
            speed_dict = dict()
        else:
            temp = np.array(list(pop[pop_idx:pop_idx+nbr_actors*nbr_vel_points]))
            temp = np.reshape(temp,(nbr_actors,nbr_vel_points))
            pop_idx += nbr_actors*nbr_vel_points
           
        for i in range(nbr_actors):
            speed_dict[i] = np.zeros(frame_durations[i]) #+1?
            delta = int(np.floor(len(speed_dict[i])/(nbr_vel_points-1)))
            for j in range(nbr_vel_points-1):
                speed_dict[i][j*delta:(j+1)*delta] = np.linspace(temp[i][j],temp[i][j+1],delta)
            speed_dict[i][j*delta:] = speed_dict[i][j*delta-1]
           
   
    if bool_dict['colors'] == True:
        #Note that bp_list should only include cars.
        if len(pop) == 0:
            temp = np.random.randint(0,256,(len(color_dict),3)) #Each row corresponds to one car.
        else:
            temp = np.array(list(pop[pop_idx:pop_idx+len(color_dict)*3]))
            temp = np.reshape(temp,(len(color_dict),3))
            pop_idx += len(color_dict)*3
           
        for idx in color_dict:
            color_dict[idx] = list(temp[idx][:])

       
    bool_array = [bool_dict['clouds'],bool_dict['precip'],bool_dict['wind'],bool_dict['sun_angle'],bool_dict['fog'],bool_dict['wetness']]
    min_limits = [0,0,0,azimuth_min,0,0]
    max_limits = [100,100,100,azimuth_max,100,100]
    weathers = ['cloudiness','precipitation','wind_intensity','sun_azimuth_angle','fog_density','wetness']
   
    idx = 0

    for b in bool_array:
        if b == True:
            if len(pop) == 0:
                temp = min_limits[idx] + (max_limits[idx]-min_limits[idx])*np.random.rand()
            else:
                temp = pop[pop_idx:pop_idx+1]
                pop_idx += 1
            weather_dict[weathers[idx]] = temp
        idx += 1

    return (starting_frames,speed_dict,color_dict,weather_dict)
       

def set_weather_params(weather_dict: dict):
    weather = carla.WeatherParameters(
        cloudiness = weather_dict['cloudiness'],
        precipitation = weather_dict['precipitation'],
        wind_intensity = weather_dict['wind_intensity'],
        sun_azimuth_angle = weather_dict['sun_azimuth_angle'],
        sun_altitude_angle = -50/15625*(weather_dict['sun_azimuth_angle']-15)*(weather_dict['sun_azimuth_angle']-265), #140 corresponds to midday.
        wetness = weather_dict['wetness'],
        fog_density = weather_dict['fog_density'],
        fog_distance = weather_dict['fog_distance'],
        fog_falloff = weather_dict['fog_falloff'],
        scattering_intensity = weather_dict['scattering_intensity']
        )
    return weather


def sensor_callback(sensor_data, sensor_queue, sensor_nbr): #Used in run_simulation to capture sensor data
    sensor_data.save_to_disk('images/%d-%.6d.jpg' % (sensor_nbr, sensor_data.frame))
    print(f"Put in queue at {sensor_data.frame}")
    sensor_queue.put((sensor_data.frame, sensor_nbr))

def run_simulation(gen_nbr, nbr_tracks, coord_dict, speed_dict, delta_seconds, delta_timestamps, starting_frames, object_classes, color_dict, world, sensor_list, sensor_queue, blueprint_library):
    #Simulates one scenario.
    
    stop = False
    is_active = np.zeros(nbr_tracks) #0 means not active yet, 1 means active and 2 means not active anymore.
    frame_inds = np.zeros(nbr_tracks)
    actor_dict = dict()
    spawn_waiting = 20 #How many frames to wait between spawning and driving.
    frame_idx_glob = 0
    
    output_inds = np.zeros(nbr_tracks, dtype = np.uint32)
    coords_gt = dict()
    speeds_gt = dict()
    timestamps_gt = dict()
    framenbrs_gt = dict()

    start_frame = world.get_snapshot().frame
    
    transx = 11
    transy = -9
    
   
    while stop == False:
        for track_idx in range(nbr_tracks):
            if is_active[track_idx] == 0 and frame_idx_glob + spawn_waiting >= starting_frames[track_idx]:
                #Spawn actor
               
                coords = coord_dict[track_idx]
                nbr_wheels = 0
                if object_classes[track_idx] == 18: #Corresponds to car
                    color = color_dict[track_idx]
                    ego_bp = blueprint_library.find('vehicle.tesla.model3')
                    ego_bp.set_attribute('color',f"{color[0]},{color[1]},{color[2]}")
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
                   

                start_trans.rotation.yaw = np.degrees(np.arctan((coords[1,1]-coords[1,0])/(coords[0,1]-coords[0,0]))) #Actor should face forward
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

            elif is_active[track_idx] == 1 and frame_idx_glob - spawn_waiting >= starting_frames[track_idx]:
                #Move actor forward
                coords = coord_dict[track_idx]
                ego_actor = actor_dict[track_idx]
                idx = int(np.ceil((frame_inds[track_idx] + 1)/round(delta_timestamps/delta_seconds)))
                current_loc = ego_actor.get_location()
                current_fv = ego_actor.get_transform().get_forward_vector()
                
                if frame_idx_glob % np.rint(delta_timestamps/delta_seconds) == 0:
                    #Save ground truth information
                    try:
                        output_idx = output_inds[track_idx] #Indexation of the ground truth.
                        coords_gt[track_idx][0,output_idx] = current_loc.x - transx
                        coords_gt[track_idx][1,output_idx] = current_loc.y - transy
                        speeds_gt[track_idx][output_idx] = ego_actor.get_velocity().length()
                        timestamps_gt[track_idx][output_idx] = frame_idx_glob*delta_seconds
                        framenbrs_gt[track_idx][output_idx] = world.get_snapshot().frame
                    except IndexError: #Increase size of the arrays
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
            while (end_frame - start_frame) % np.rint(delta_timestamps/delta_seconds) != 0: #To keep everything in sync. 
                world.tick()
                end_frame += 1

        else: #Write frame numbers to a text document according to Viscando's wishes.
            filename = f"output_scenario/scenario_{gen_nbr}.txt"
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

            
            if frame_idx_glob % np.rint(delta_timestamps/delta_seconds) == 0: #Capture sensor data
                try:
                    for _ in range(len(sensor_list)):
                        s_frame = sensor_queue.get(True,10.0) 
                except Empty:
                    print(f"Some data was lost, {world.get_snapshot().frame}")
            
            frame_idx_glob += 1


    print(frame_idx_glob)
     
    #Saving the ground truth to a csv file.
    gt_output = np.zeros((np.sum(output_inds),7))
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

    filename = f"ground_truth/ground_truth_{gen_nbr}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, sep = ';', index = None)
    #Saving the ground truth to a csv file.
    
    
