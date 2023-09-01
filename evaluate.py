

from simulation_methods import compute_errors
from simulation_main import get_param_values
import matplotlib.pyplot as plt
import numpy as np

def main():

    """
    This is code that was used to examine and evaluate the results in different ways.
    It is not relevant to the generation and running of simulated scenarios.
    It is quite unstructured and has few comments, but it is still included for reference.
    """

    op_path = 'C:/Users/elias/outputs_viscando' #Path to OTUS3D estimations
    gt_path = 'C:/Users/elias/ground_truth' #Path to ground truth
    
    nbr_batches = 4
    nbr_gens = 3
    pop_size = 20
    nbr_actors = 7
    delta_timestamps = 0.05

    nbr_points = nbr_gens*nbr_batches*pop_size 

    alt_values = np.zeros(nbr_points)
    cl_values = np.zeros(nbr_points)
    dist_errors = np.zeros(nbr_points)
    speed_errors = np.zeros(nbr_points)
    perc_errors = np.zeros(nbr_points)
    az_values = np.zeros(nbr_points)

    alt_values_cars = np.zeros(int(nbr_points/2)) #altitude values for the car scenarios
    cl_values_cars = np.zeros(int(nbr_points/2))
    alt_values_bikes = np.zeros(int(nbr_points/2))
    cl_values_bikes = np.zeros(int(nbr_points/2))
    az_values_cars = np.zeros(int(nbr_points/2))
    az_values_bikes = np.zeros(int(nbr_points/2))

    luminance_values = np.zeros((int(nbr_points/2),3))

    track_errors_dist = np.zeros(10)#There are ten tracks in total. The mean errors of each track. Bike tracks at the end.
    track_errors_speed = np.zeros(10)
    track_errors_perc = np.zeros(10)
    #step_perc_avg = np.zeros(10)

    error_values_dist_car = np.zeros((int(nbr_points/2), nbr_actors)) #Contains all errors in the car scenarios. Each row is one scenario and each column
    #is one trajectory.
    error_values_speed_car = np.zeros((int(nbr_points/2), nbr_actors))
    error_values_perc_car = np.zeros((int(nbr_points/2), nbr_actors))
    step_values_car = np.zeros((int(nbr_points/2), nbr_actors))

    error_values_dist_bike = np.zeros((int(nbr_points/2), nbr_actors))
    error_values_speed_bike = np.zeros((int(nbr_points/2), nbr_actors))
    error_values_perc_bike = np.zeros((int(nbr_points/2), nbr_actors))
    step_values_bike = np.zeros((int(nbr_points/2), nbr_actors))

    error_values_dist_all = np.zeros((int(nbr_points), nbr_actors))
    error_values_speed_all = np.zeros((int(nbr_points), nbr_actors))
    error_values_perc_all = np.zeros((int(nbr_points), nbr_actors))
    step_values_all = np.zeros((int(nbr_points), nbr_actors))

    alt_ctr = 0

    ts_idx = []

    for current_gen in range(1,nbr_gens+1):
        for batch in range(nbr_batches):
            pop = get_param_values(batch, pop_size, delta_timestamps, current_gen, nbr_actors, gt_path)
            #(avg_errors, dist_results, speed_results, perc_misclass, step_perc, coord_diff, speed_diff, class_diff, coords_gt_save) = compute_errors(batch, nbr_actors, pop_size, current_gen, True, op_path, gt_path)
            (avg_errors, dist_results, speed_results, perc_misclass, ts_save) = compute_errors(batch, nbr_actors, pop_size, current_gen, False, op_path, gt_path)
            #avg_errors is of size 20X3 and the others of size 20X7
            for idx in range(len(pop)):
                if (batch == 0 or batch == 1) and ts_save[idx][0]-ts_save[idx][1] < 2.5 and ts_save[idx][0]-ts_save[idx][1] > 0: #Occlusion event
                #if (batch == 2 or batch == 3) and ts_save[idx][0]-ts_save[idx][2] < 1 and ts_save[idx][0]-ts_save[idx][2] > 0:
                #if (batch == 2 or batch == 3) and ts_save[idx][2]-ts_save[idx][1] < 5 and ts_save[idx][2]-ts_save[idx][1] > 3.5:
                #if (batch == 0 or batch == 1) and ts_save[idx][3]-ts_save[idx][1] < 4 and ts_save[idx][3]-ts_save[idx][1] > 2:
                    print((current_gen-1,batch,idx))
                    ts_idx.append([current_gen-1,batch,idx])
                arr_idx = (current_gen-1)*nbr_batches*pop_size + batch*pop_size+idx
                
                alt_values[arr_idx] = -50/15625*(pop[idx][-2]-15)*(pop[idx][-2]-265) #25<az_angle<255
                cl_values[arr_idx] = pop[idx][-1]
                dist_errors[arr_idx] = avg_errors[idx][0]
                az_values[arr_idx] = pop[idx][-2]
                #temp = [np.mean(np.abs(x[1])) for x in coord_diff[idx]]
                #print(temp)
                #dist_errors[arr_idx] = np.mean(temp)
                
                speed_errors[arr_idx] = avg_errors[idx][1]
                perc_errors[arr_idx] = avg_errors[idx][2]

                #error_values_dist_all[arr_idx] = temp
                error_values_dist_all[arr_idx] = dist_results[idx]
                error_values_speed_all[arr_idx] = speed_results[idx]
                error_values_perc_all[arr_idx] = perc_misclass[idx]
                #step_values_all[arr_idx] = step_perc[idx]

                """
                if cl_values[arr_idx] > 80 and alt_values[arr_idx] < 20:
                    print(f'({current_gen-1}, {batch}, {idx}):')
                    print(alt_values[arr_idx])
                    print(cl_values[arr_idx])
                    print(dist_results[idx])
                    print(perc_misclass[idx])
                    alt_ctr += 1
                """

                if batch == 1 or batch == 0: #car trajectories
                    arr_idx = int((current_gen-1)*nbr_batches*pop_size/2 + batch*pop_size+idx)
                    #print(arr_idx)
                    error_values_dist_car[arr_idx,:] = dist_results[idx]
                    #error_values_dist_car[arr_idx,:] = temp
                    error_values_speed_car[arr_idx,:] = speed_results[idx]
                    error_values_perc_car[arr_idx,:] = perc_misclass[idx]
                    alt_values_cars[arr_idx] = -50/15625*(pop[idx][-2]-15)*(pop[idx][-2]-265) 
                    cl_values_cars[arr_idx] = pop[idx][-1]
                    #step_values_car[arr_idx] = step_perc[idx]
                    az_values_cars[arr_idx] = pop[idx][-2]
                    
                    colors = pop[idx][nbr_actors:-2]
                    colors = [val/255 for val in colors]
                    colors_lin = [val/12.92 if val <= 0.04045 else ((val + 0.055)/1.055)**2.4 for val in colors]
                    for i in range(0,9,3):
                        luminance_values[arr_idx,int(i/3)] = 0.2126 * colors_lin[i] + 0.7152 * colors_lin[i+1] + 0.0722 * colors_lin[i+2]
                    #print(luminance_values[arr_idx,:])

                    """
                    if (cl_values[arr_idx] > 80 or alt_values[arr_idx] < 20) and np.min(luminance_values[arr_idx,:]) < 0.1:
                        print(f'({current_gen-1}, {batch}, {idx}):')
                        print(alt_values[arr_idx])
                        print(cl_values[arr_idx])
                        print(luminance_values[arr_idx,:])
                        print(dist_results[idx])
                        print(perc_misclass[idx])
                        alt_ctr += 1
                    """
                
                    """
                    error_values_dist_car.append(dist_results[idx])
                    error_values_speed_car.append(speed_results[idx])
                    error_values_perc_car.append(perc_misclass[idx])
                    """
                    

                else: #bike trajectories
                    arr_idx = int((current_gen-1)*nbr_batches*pop_size/2 + (batch-2)*pop_size+idx)
                    error_values_dist_bike[arr_idx,:] = dist_results[idx]
                    #error_values_dist_bike[arr_idx,:] = temp
                    error_values_speed_bike[arr_idx,:] = speed_results[idx]
                    error_values_perc_bike[arr_idx,:] = perc_misclass[idx]
                    alt_values_bikes[arr_idx] = -50/15625*(pop[idx][-2]-15)*(pop[idx][-2]-265)
                    cl_values_bikes[arr_idx] = pop[idx][-1]
                    #step_values_bike[arr_idx] = step_perc[idx]
                    az_values_bikes[arr_idx] = pop[idx][-2]
                    """
                    error_values_dist_bike.append(dist_results[idx])
                    error_values_speed_bike.append(speed_results[idx])
                    error_values_perc_bike.append(perc_misclass[idx])
                    """
                for idx2 in range(len(dist_results[idx])):
                    idx_save = idx2
                    if (batch == 2 or batch == 3) and idx2 < 3: #This way, the bike tracks are put at the end of the track_error arrays
                        idx2 += 7
                    
                    track_errors_dist[idx2] += dist_results[idx][idx_save]
                    #track_errors_dist[idx2] += temp[idx_save]
                    track_errors_speed[idx2] += speed_results[idx][idx_save]
                    track_errors_perc[idx2] += perc_misclass[idx][idx_save]
                    #step_perc_avg[idx2] += step_perc[idx][idx_save]


                    idx2 = idx_save
                    #Add each array of length 10 to one long list. Then go through this list. 

    track_errors_dist[0:3] = track_errors_dist[0:3]/(0.5*nbr_points) #Halved because only half of the scenarios had cars in them
    track_errors_speed[0:3] = track_errors_speed[0:3]/(0.5*nbr_points)
    track_errors_perc[0:3] = track_errors_perc[0:3]/(0.5*nbr_points)
    #step_perc_avg[0:3] = step_perc_avg[0:3]/(0.5*nbr_points)

    track_errors_dist[3:7] = track_errors_dist[3:7]/nbr_points
    track_errors_speed[3:7] = track_errors_speed[3:7]/nbr_points
    track_errors_perc[3:7] = track_errors_perc[3:7]/nbr_points
    #step_perc_avg[3:7] = step_perc_avg[3:7]/nbr_points

    track_errors_dist[7:10] = track_errors_dist[7:10]/(0.5*nbr_points)
    track_errors_speed[7:10] = track_errors_speed[7:10]/(0.5*nbr_points)
    track_errors_perc[7:10] = track_errors_perc[7:10]/(0.5*nbr_points)
    #step_perc_avg[7:10] = step_perc_avg[7:10]/(0.5*nbr_points)

    track_variances_dist = np.zeros(10)
    track_variances_speed = np.zeros(10)
    track_variances_perc = np.zeros(10)

    for idx1 in range(len(error_values_dist_car)):
        for idx2 in range(len(error_values_dist_car[idx1])):
            track_variances_dist[idx2] += (track_errors_dist[idx2]-error_values_dist_car[idx1][idx2])**2
            track_variances_speed[idx2] += (track_errors_speed[idx2]-error_values_speed_car[idx1][idx2])**2
            track_variances_perc[idx2] += (track_errors_perc[idx2]-error_values_perc_car[idx1][idx2])**2

    for idx1 in range(len(error_values_dist_bike)):
        for idx2 in range(len(error_values_dist_bike[idx1])):
            idx_save = idx2
            if idx2 < 3:
                idx2 += 7
            track_variances_dist[idx2] += (track_errors_dist[idx2]-error_values_dist_bike[idx1][idx_save])**2
            track_variances_speed[idx2] += (track_errors_speed[idx2]-error_values_speed_bike[idx1][idx_save])**2
            track_variances_perc[idx2] += (track_errors_perc[idx2]-error_values_perc_bike[idx1][idx_save])**2

            idx2 = idx_save

    track_variances_dist[0:3] = track_variances_dist[0:3]/(0.5*nbr_points-1)
    track_variances_speed[0:3] = track_variances_speed[0:3]/(0.5*nbr_points-1)
    track_variances_perc[0:3] = track_variances_perc[0:3]/(0.5*nbr_points-1)

    track_variances_dist[3:7] = track_variances_dist[3:7]/(nbr_points-1)
    track_variances_speed[3:7] = track_variances_speed[3:7]/(nbr_points-1)
    track_variances_perc[3:7] = track_variances_perc[3:7]/(nbr_points-1)

    track_variances_dist[7:10] = track_variances_dist[7:10]/(0.5*nbr_points-1)
    track_variances_speed[7:10] = track_variances_speed[7:10]/(0.5*nbr_points-1)
    track_variances_perc[7:10] = track_variances_perc[7:10]/(0.5*nbr_points-1)
    print("Distance errors:")
    print(track_errors_dist)
    print("Speed errors:")
    print(track_errors_speed)
    print("Percentage misclassified:")
    print(track_errors_perc)

    conf_ints_dist = np.zeros(10)
    conf_ints_speed = np.zeros(10)
    conf_ints_perc = np.zeros(10)

    conf_ints_dist[0:3] = 1.96*np.sqrt(track_variances_dist[0:3]/(0.5*nbr_points))
    conf_ints_speed[0:3] = 1.96*np.sqrt(track_variances_speed[0:3]/(0.5*nbr_points))
    conf_ints_perc[0:3] = 1.96*np.sqrt(track_variances_perc[0:3]/(0.5*nbr_points))

    conf_ints_dist[3:7] = 1.96*np.sqrt(track_variances_dist[3:7]/nbr_points)
    conf_ints_speed[3:7] = 1.96*np.sqrt(track_variances_speed[3:7]/nbr_points)
    conf_ints_perc[3:7] = 1.96*np.sqrt(track_variances_perc[3:7]/nbr_points)

    conf_ints_dist[7:10] = 1.96*np.sqrt(track_variances_dist[7:10]/(0.5*nbr_points))
    conf_ints_speed[7:10] = 1.96*np.sqrt(track_variances_speed[7:10]/(0.5*nbr_points))
    conf_ints_perc[7:10] = 1.96*np.sqrt(track_variances_perc[7:10]/(0.5*nbr_points))

    print("Confidence intervals:")
    print('-----')
    print(conf_ints_dist)
    print(conf_ints_speed)
    print(conf_ints_perc)
    print('-----')
    

    corr_alt_dist = np.corrcoef(alt_values,dist_errors) #Correlation between the mean of all distance errors in a scenario and the sun altitude value.
    corr_alt_speed = np.corrcoef(alt_values,speed_errors) #Do this between alt_values and the errors in each of the trajectories.
    corr_alt_perc = np.corrcoef(alt_values,perc_errors)

    corr_cloud_dist = np.corrcoef(cl_values,dist_errors)
    corr_cloud_speed = np.corrcoef(cl_values,speed_errors)
    corr_cloud_perc = np.corrcoef(cl_values,perc_errors)

    print("Correlation sun altitude/distance errors:")
    print(corr_alt_dist)
    print("Correlation sun altitude/speed errors:")
    print(corr_alt_speed)
    print("Correlation sun altitude/percentgae misclassified:")
    print(corr_alt_perc)
    print("Correlation cloudiness/distance errors:")
    print(corr_cloud_dist)
    print("Correlation cloudiness/speed errors:")
    print(corr_cloud_speed)
    print("Correlation cloudiness/percentage misclassified:")
    print(corr_cloud_perc)

    corr_alt_dist_c1 = np.corrcoef(alt_values_cars, error_values_dist_car[:,0])
    corr_alt_dist_c2 = np.corrcoef(alt_values_cars, error_values_dist_car[:,1])
    corr_alt_dist_c3 = np.corrcoef(alt_values_cars, error_values_dist_car[:,2])

    corr_alt_dist_b1 = np.corrcoef(alt_values_bikes, error_values_dist_bike[:,0])
    corr_alt_dist_b2 = np.corrcoef(alt_values_bikes, error_values_dist_bike[:,1])
    corr_alt_dist_b3 = np.corrcoef(alt_values_bikes, error_values_dist_bike[:,2])

    corr_alt_dist_w1 = np.corrcoef(alt_values, error_values_dist_all[:,3])
    corr_alt_dist_w2 = np.corrcoef(alt_values, error_values_dist_all[:,4])
    corr_alt_dist_w3 = np.corrcoef(alt_values, error_values_dist_all[:,5])
    corr_alt_dist_w4 = np.corrcoef(alt_values, error_values_dist_all[:,6])

    print('---alt/dists---')
    print(corr_alt_dist_c1)
    print(corr_alt_dist_c2)
    print(corr_alt_dist_c3)
    print(corr_alt_dist_b1)
    print(corr_alt_dist_b2)
    print(corr_alt_dist_b3)
    print(corr_alt_dist_w1)
    print(corr_alt_dist_w2)
    print(corr_alt_dist_w3)
    print(corr_alt_dist_w4)
    
    corr_alt_speed_c1 = np.corrcoef(alt_values_cars, error_values_speed_car[:,0])
    corr_alt_speed_c2 = np.corrcoef(alt_values_cars, error_values_speed_car[:,1])
    corr_alt_speed_c3 = np.corrcoef(alt_values_cars, error_values_speed_car[:,2])

    corr_alt_speed_b1 = np.corrcoef(alt_values_bikes, error_values_speed_bike[:,0])
    corr_alt_speed_b2 = np.corrcoef(alt_values_bikes, error_values_speed_bike[:,1])
    corr_alt_speed_b3 = np.corrcoef(alt_values_bikes, error_values_speed_bike[:,2])

    corr_alt_speed_w1 = np.corrcoef(alt_values, error_values_speed_all[:,3])
    corr_alt_speed_w2 = np.corrcoef(alt_values, error_values_speed_all[:,4])
    corr_alt_speed_w3 = np.corrcoef(alt_values, error_values_speed_all[:,5])
    corr_alt_speed_w4 = np.corrcoef(alt_values, error_values_speed_all[:,6])
    print('---alt/speeds---')
    print(corr_alt_speed_c1)
    print(corr_alt_speed_c2)
    print(corr_alt_speed_c3)
    print(corr_alt_speed_b1)
    print(corr_alt_speed_b2)
    print(corr_alt_speed_b3)
    print(corr_alt_speed_w1)
    print(corr_alt_speed_w2)
    print(corr_alt_speed_w3)
    print(corr_alt_speed_w4)

    corr_cl_dist_c1 = np.corrcoef(cl_values_cars, error_values_dist_car[:,0])
    corr_cl_dist_c2 = np.corrcoef(cl_values_cars, error_values_dist_car[:,1])
    corr_cl_dist_c3 = np.corrcoef(cl_values_cars, error_values_dist_car[:,2])

    corr_cl_dist_b1 = np.corrcoef(cl_values_bikes, error_values_dist_bike[:,0])
    corr_cl_dist_b2 = np.corrcoef(cl_values_bikes, error_values_dist_bike[:,1])
    corr_cl_dist_b3 = np.corrcoef(cl_values_bikes, error_values_dist_bike[:,2])

    corr_cl_dist_w1 = np.corrcoef(cl_values, error_values_dist_all[:,3])
    corr_cl_dist_w2 = np.corrcoef(cl_values, error_values_dist_all[:,4])
    corr_cl_dist_w3 = np.corrcoef(cl_values, error_values_dist_all[:,5])
    corr_cl_dist_w4 = np.corrcoef(cl_values, error_values_dist_all[:,6])
    print('---cl/dists---')
    print(corr_cl_dist_c1)
    print(corr_cl_dist_c2)
    print(corr_cl_dist_c3)
    print(corr_cl_dist_b1)
    print(corr_cl_dist_b2)
    print(corr_cl_dist_b3)
    print(corr_cl_dist_w1)
    print(corr_cl_dist_w2)
    print(corr_cl_dist_w3)
    print(corr_cl_dist_w4)

    corr_cl_speed_c1 = np.corrcoef(cl_values_cars, error_values_speed_car[:,0])
    corr_cl_speed_c2 = np.corrcoef(cl_values_cars, error_values_speed_car[:,1])
    corr_cl_speed_c3 = np.corrcoef(cl_values_cars, error_values_speed_car[:,2])

    corr_cl_speed_b1 = np.corrcoef(cl_values_bikes, error_values_speed_bike[:,0])
    corr_cl_speed_b2 = np.corrcoef(cl_values_bikes, error_values_speed_bike[:,1])
    corr_cl_speed_b3 = np.corrcoef(cl_values_bikes, error_values_speed_bike[:,2])

    corr_cl_speed_w1 = np.corrcoef(cl_values, error_values_speed_all[:,3])
    corr_cl_speed_w2 = np.corrcoef(cl_values, error_values_speed_all[:,4])
    corr_cl_speed_w3 = np.corrcoef(cl_values, error_values_speed_all[:,5])
    corr_cl_speed_w4 = np.corrcoef(cl_values, error_values_speed_all[:,6])
    print('---cl/speeds---')
    print(corr_cl_speed_c1)
    print(corr_cl_speed_c2)
    print(corr_cl_speed_c3)
    print(corr_cl_speed_b1)
    print(corr_cl_speed_b2)
    print(corr_cl_speed_b3)
    print(corr_cl_speed_w1)
    print(corr_cl_speed_w2)
    print(corr_cl_speed_w3)
    print(corr_cl_speed_w4)
    
    corr_alt_perc_c1 = np.corrcoef(alt_values_cars, error_values_perc_car[:,0])
    corr_alt_perc_c2 = np.corrcoef(alt_values_cars, error_values_perc_car[:,1])
    corr_alt_perc_c3 = np.corrcoef(alt_values_cars, error_values_perc_car[:,2])

    corr_alt_perc_b1 = np.corrcoef(alt_values_bikes, error_values_perc_bike[:,0])
    corr_alt_perc_b2 = np.corrcoef(alt_values_bikes, error_values_perc_bike[:,1])
    corr_alt_perc_b3 = np.corrcoef(alt_values_bikes, error_values_perc_bike[:,2])

    corr_alt_perc_w1 = np.corrcoef(alt_values, error_values_perc_all[:,3])
    corr_alt_perc_w2 = np.corrcoef(alt_values, error_values_perc_all[:,4])
    corr_alt_perc_w3 = np.corrcoef(alt_values, error_values_perc_all[:,5])
    corr_alt_perc_w4 = np.corrcoef(alt_values, error_values_perc_all[:,6])

    print('---alt/perc---')
    print(corr_alt_perc_c1)
    print(corr_alt_perc_c2)
    print(corr_alt_perc_c3)
    print(corr_alt_perc_b1)
    print(corr_alt_perc_b2)
    print(corr_alt_perc_b3)
    print(corr_alt_perc_w1)
    print(corr_alt_perc_w2)
    print(corr_alt_perc_w3)
    print(corr_alt_perc_w4)

    corr_cl_perc_c1 = np.corrcoef(cl_values_cars, error_values_perc_car[:,0])
    corr_cl_perc_c2 = np.corrcoef(cl_values_cars, error_values_perc_car[:,1])
    corr_cl_perc_c3 = np.corrcoef(cl_values_cars, error_values_perc_car[:,2])

    corr_cl_perc_b1 = np.corrcoef(cl_values_bikes, error_values_perc_bike[:,0])
    corr_cl_perc_b2 = np.corrcoef(cl_values_bikes, error_values_perc_bike[:,1])
    corr_cl_perc_b3 = np.corrcoef(cl_values_bikes, error_values_perc_bike[:,2])

    corr_cl_perc_w1 = np.corrcoef(cl_values, error_values_perc_all[:,3])
    corr_cl_perc_w2 = np.corrcoef(cl_values, error_values_perc_all[:,4])
    corr_cl_perc_w3 = np.corrcoef(cl_values, error_values_perc_all[:,5])
    corr_cl_perc_w4 = np.corrcoef(cl_values, error_values_perc_all[:,6])

    print('---cl/perc---')
    print(corr_cl_perc_c1)
    print(corr_cl_perc_c2)
    print(corr_cl_perc_c3)
    print(corr_cl_perc_b1)
    print(corr_cl_perc_b2)
    print(corr_cl_perc_b3)
    print(corr_cl_perc_w1)
    print(corr_cl_perc_w2)
    print(corr_cl_perc_w3)
    print(corr_cl_perc_w4)

    corr_col_dist_1 = np.corrcoef(luminance_values[:,0],error_values_dist_car[:,0])
    corr_col_dist_2 = np.corrcoef(luminance_values[:,1],error_values_dist_car[:,1])
    corr_col_dist_3 = np.corrcoef(luminance_values[:,2],error_values_dist_car[:,2])

    corr_col_speed_1 = np.corrcoef(luminance_values[:,0],error_values_speed_car[:,0])
    corr_col_speed_2 = np.corrcoef(luminance_values[:,1],error_values_speed_car[:,1])
    corr_col_speed_3 = np.corrcoef(luminance_values[:,2],error_values_speed_car[:,2])

    corr_col_perc_1 = np.corrcoef(luminance_values[:,0],error_values_perc_car[:,0])
    corr_col_perc_2 = np.corrcoef(luminance_values[:,1],error_values_perc_car[:,1])
    corr_col_perc_3 = np.corrcoef(luminance_values[:,2],error_values_perc_car[:,2])


    print('---col---')
    print(corr_col_dist_1)
    print(corr_col_dist_2)
    print(corr_col_dist_3)
    print(corr_col_speed_1)
    print(corr_col_speed_2)
    print(corr_col_speed_3)
    print(corr_col_perc_1)
    print(corr_col_perc_2)
    print(corr_col_perc_3)

    print(alt_ctr)

    """ 
    gen_find = 0
    batch_find = 0
    ind_find = 12
    arr_idx = gen_find*nbr_batches*pop_size + batch_find*pop_size+ind_find

    print(error_values_dist_all[arr_idx])
    print(error_values_speed_all[arr_idx])
    print(error_values_perc_all[arr_idx])

    print("nbr of steps out of the total:")
    #print(step_perc_avg)
    """
    """
    print('Correlation nbr points/alt')
    print(np.corrcoef(alt_values_cars, step_values_car[:,0]))
    print(np.corrcoef(alt_values_cars, step_values_car[:,1]))
    print(np.corrcoef(alt_values_cars, step_values_car[:,2]))
    print(np.corrcoef(alt_values_bikes, step_values_bike[:,0]))
    print(np.corrcoef(alt_values_bikes, step_values_bike[:,1]))
    print(np.corrcoef(alt_values_bikes, step_values_bike[:,2]))
    print(np.corrcoef(alt_values, step_values_all[:,3]))
    print(np.corrcoef(alt_values, step_values_all[:,4]))
    print(np.corrcoef(alt_values, step_values_all[:,5]))
    print(np.corrcoef(alt_values, step_values_all[:,6]))
    """
    for i in range(3):
        print(f"{i}:")
        dist_sort = np.argsort(error_values_dist_car[:,i])[-6:-1]
        speed_sort = np.argsort(error_values_speed_car[:,i])[-6:-1]
        perc_sort = np.argsort(error_values_perc_car[:,i])[-6:-1]
        print(dist_sort)
        print(speed_sort)
        print(perc_sort)
        #print(alt_values_cars[dist_sort])
        #print(cl_values_cars[dist_sort])
        print(error_values_dist_car[dist_sort,i])
        print(error_values_speed_car[speed_sort,i])
        print(error_values_perc_car[perc_sort,i])

    for i in range(3,7):
        print(f"{i}:")
        dist_sort = np.argsort(error_values_dist_all[:,i])[-6:-1]
        speed_sort = np.argsort(error_values_speed_all[:,i])[-6:-1]
        perc_sort = np.argsort(error_values_perc_all[:,i])[-6:-1]
        print(dist_sort)
        print(speed_sort)
        print(perc_sort)
        print(error_values_dist_all[dist_sort,i])
        print(error_values_speed_all[speed_sort,i])
        print(error_values_perc_all[perc_sort,i])

    for i in range(3):
        print(f"{i}:")
        dist_sort = np.argsort(error_values_dist_bike[:,i])[-6:-1]
        speed_sort = np.argsort(error_values_speed_bike[:,i])[-6:-1]
        perc_sort = np.argsort(error_values_perc_bike[:,i])[-6:-1]
        print(dist_sort)
        print(speed_sort)
        print(perc_sort)
        print(error_values_dist_bike[dist_sort,i])
        print(error_values_speed_bike[speed_sort,i])
        print(error_values_perc_bike[perc_sort,i])
        
    ts_error_dist = np.zeros(7)
    ts_error_speed = np.zeros(7)
    ts_error_perc = np.zeros(7)

    for idx in ts_idx:
        current_gen = idx[0]
        batch = idx[1]
        ind_idx = idx[2]
        arr_idx = current_gen*nbr_batches*pop_size + batch*pop_size+ind_idx
        ts_error_dist += error_values_dist_all[arr_idx,:]
        ts_error_speed += error_values_speed_all[arr_idx,:]
        ts_error_perc += error_values_perc_all[arr_idx,:]

    ts_mean_dist = ts_error_dist/len(ts_idx)
    ts_mean_speed = ts_error_speed/len(ts_idx)
    ts_mean_perc = ts_error_perc/len(ts_idx)
    

    print(ts_mean_dist)
    print(ts_mean_speed)
    print(ts_mean_perc)
    print(len(ts_idx))

    #print(np.sum(error_values_dist_car[:,0]<ts_mean_dist[0])/(0.5*nbr_points))
    #print(np.sum(error_values_speed_car[:,0]<ts_mean_speed[0])/(0.5*nbr_points))
    #print(np.sum(error_values_perc_car[:,0]<ts_mean_perc[0])/(0.5*nbr_points))

    print(np.sum(error_values_dist_bike[:,2]<ts_mean_dist[2])/(0.5*nbr_points))
    print(np.sum(error_values_speed_bike[:,2]<ts_mean_speed[2])/(0.5*nbr_points))
    print(np.sum(error_values_perc_bike[:,2]<ts_mean_perc[2])/(0.5*nbr_points))

    #print(np.sum(error_values_dist_all[:,3]<ts_mean_dist[0])/(nbr_points)) ts_mean_dist[3]?
    #print(np.sum(error_values_speed_all[:,3]<ts_mean_speed[0])/(nbr_points))
    #print(np.sum(error_values_perc_all[:,3]<ts_mean_perc[0])/(nbr_points))
    print('---')
    az_idx = 5
    az_min = 150
    az_max = 200
    az_array = error_values_dist_all[:,az_idx]
    sub_array1 = az_array[az_values>az_min]
    sub_array2 = az_array[az_values<az_max]
    sub_array3 = az_array[az_values<az_min]
    sub_array4 = az_array[az_values>az_max]
    print(np.mean(np.concatenate((sub_array1,sub_array2))))
    print(np.mean(np.concatenate((sub_array3,sub_array4))))


    az_idx = 2
    az_min = 80
    az_max = 140
    az_array = error_values_dist_car[:,az_idx]
    sub_array1 = az_array[az_values_cars>az_min]
    sub_array2 = az_array[az_values_cars<az_max]
    sub_array3 = az_array[az_values_cars<az_min]
    sub_array4 = az_array[az_values_cars>az_max]
    print(np.mean(np.concatenate((sub_array1,sub_array2))))
    print(np.mean(np.concatenate((sub_array3,sub_array4))))

    az_idx = 2
    az_min = 200
    az_max = 255
    az_array = error_values_perc_bike[:,az_idx]
    sub_array1 = az_array[az_values_bikes>az_min]
    sub_array2 = az_array[az_values_bikes<az_max]
    sub_array3 = az_array[az_values_bikes<az_min]
    sub_array4 = az_array[az_values_bikes>az_max]
    print(np.mean(np.concatenate((sub_array1,sub_array2))))
    print(np.mean(np.concatenate((sub_array3,sub_array4))))

    #print(np.sum(error_values_dist_bike[:,2]<ts_mean_dist[2])/(0.5*nbr_points))
    #print(np.sum(error_values_speed_bike[:,2]<ts_mean_speed[2])/(0.5*nbr_points))
    #print(np.sum(error_values_perc_bike[:,2]<ts_mean_perc[2])/(0.5*nbr_points))

    #print(np.sum(error_values_dist_car[:,0]<ts_mean_dist[0])/(0.5*nbr_points))
    #print(np.sum(error_values_speed_car[:,0]<ts_mean_speed[0])/(0.5*nbr_points))
    #print(np.sum(error_values_perc_car[:,0]<ts_mean_perc[0])/(0.5*nbr_points))

    

    error_values_dist_car_all = (error_values_dist_car[:,0] + error_values_dist_car[:,1] + error_values_dist_car[:,2])/3
    error_values_dist_bike_all = (error_values_dist_bike[:,0] + error_values_dist_bike[:,1] + error_values_dist_bike[:,2])/3
    error_values_dist_walker_all = (error_values_dist_all[:,3] + error_values_dist_all[:,4] + error_values_dist_all[:,5] + error_values_dist_all[:,6])/4

    #print(np.corrcoef(alt_values_cars,error_values_dist_car_all))
    #print(np.corrcoef(alt_values_bikes,error_values_dist_bike_all))
    #print(np.corrcoef(alt_values,error_values_dist_walker_all))

    error_dist_walker_car = (error_values_dist_car[:,3] + error_values_dist_car[:,4] + error_values_dist_car[:,5] + error_values_dist_car[:,6])/4
    error_dist_walker_bike = (error_values_dist_bike[:,3] + error_values_dist_bike[:,4] + error_values_dist_bike[:,5] + error_values_dist_bike[:,6])/4
    error_speed_walker_car = (error_values_speed_car[:,3] + error_values_speed_car[:,4] + error_values_speed_car[:,5] + error_values_speed_car[:,6])/4
    error_speed_walker_bike = (error_values_speed_bike[:,3] + error_values_speed_bike[:,4] + error_values_speed_bike[:,5] + error_values_speed_bike[:,6])/4
    error_perc_walker_car = (error_values_perc_car[:,3] + error_values_perc_car[:,4] + error_values_perc_car[:,5] + error_values_perc_car[:,6])/4
    error_perc_walker_bike = (error_values_perc_bike[:,3] + error_values_perc_bike[:,4] + error_values_perc_bike[:,5] + error_values_perc_bike[:,6])/4

    print('---')
    print(np.mean(error_dist_walker_car))
    print(np.mean(error_dist_walker_bike))
    print(np.mean(error_speed_walker_car))
    print(np.mean(error_speed_walker_bike))
    print(np.mean(error_perc_walker_car))
    print(np.mean(error_perc_walker_bike))

    dist_all_w = np.concatenate((error_values_dist_all[:,3],error_values_dist_all[:,4], error_values_dist_all[:,5], error_values_dist_all[:,6]))
    dist_all_c = np.concatenate((error_values_dist_car[:,0],error_values_dist_car[:,1], error_values_dist_car[:,2]))
    alt_all_w = np.concatenate((alt_values, alt_values, alt_values, alt_values))
    alt_all_c = np.concatenate((alt_values_cars, alt_values_cars, alt_values_cars))
    print(np.corrcoef(alt_all_w, dist_all_w))
    print(np.corrcoef(alt_all_c, dist_all_c))

    """
    plt.figure()
    #plt.subplot(311)
    plt.plot(alt_values_cars,error_values_dist_car_all,'.')
    plt.title('Sun altitude and distance errors: Car trajectories')
    plt.xlabel('altitude values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.show()
    #plt.subplot(312)
    plt.plot(alt_values_bikes,error_values_dist_bike_all,'.')
    plt.title('Sun altitude and distance errors: Bicycle trajectories')
    plt.xlabel('altitude values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.show()
    #plt.subplot(313)
    plt.plot(alt_values,error_values_dist_walker_all,'.')
    plt.title('Sun altitude and distance errors: Walker trajectories')
    plt.xlabel('altitude values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.show()
    """
    """
    plt.figure()
    #plt.subplot(311)
    plt.plot(cl_values_cars,error_values_dist_car_all,'.')
    plt.title('Cloudiness and distance errors: Car trajectories')
    plt.xlabel('cloudiness values')
    plt.ylabel('distance errors (m)')
    plt.show()
    #plt.subplot(312)
    plt.plot(cl_values_bikes,error_values_dist_bike_all,'.')
    plt.title('Cloudiness and distance errors: Bicycle trajectories')
    plt.xlabel('cloudiness values')
    plt.ylabel('distance errors (m)')
    plt.show()
    #plt.subplot(313)
    plt.plot(cl_values,error_values_dist_walker_all,'.')
    plt.title('Cloudiness and distance errors: Walker trajectories')
    plt.xlabel('cloudiness values')
    plt.ylabel('distance errors (m)')
    plt.show()
    """
    """
    lum_all = np.concatenate((luminance_values[:,0], luminance_values[:,1], luminance_values[:,2]))
    dist_all = np.concatenate((error_values_dist_car[:,0],error_values_dist_car[:,1], error_values_dist_car[:,2]))
    speed_all = np.concatenate((error_values_speed_car[:,0],error_values_speed_car[:,1], error_values_speed_car[:,2]))
    perc_all = np.concatenate((error_values_perc_car[:,0],error_values_perc_car[:,1], error_values_perc_car[:,2]))

    
    print(np.corrcoef(lum_all,dist_all))
    print(np.corrcoef(lum_all,speed_all))
    print(np.corrcoef(lum_all,perc_all))      

    
    plt.figure()
    plt.plot(lum_all, np.concatenate((error_values_dist_car[:,0],error_values_dist_car[:,1], error_values_dist_car[:,2])),'.')
    plt.xlabel('luminance values')
    plt.ylabel('distance errors (m)')
    plt.title('Luminance values and distance errors')
    plt.show()
    """

    
    
    plt.figure()
    #plt.subplot(411)
    plt.plot(az_values,error_values_dist_all[:,3],'.')
    plt.xlabel('azimuth values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.show()
    plt.subplot(312)
    plt.plot(az_values,error_values_dist_all[:,4],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    
    plt.plot(az_values,error_values_dist_all[:,5],'.')
    plt.xlabel('azimuth values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.title('Sun azimuth and distance errors: Walker trajectory 2')
    plt.show()
    
    #plt.plot(az_values,error_values_dist_all[:,6],'.')
    #plt.xlabel('altitude values')
    #plt.ylabel('distance errors')
    #plt.show()
    
    """
    plt.figure()
    plt.subplot(411)
    plt.plot(az_values,error_values_speed_all[:,3],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values,error_values_speed_all[:,4],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(az_values,error_values_speed_all[:,5],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    plt.plot(az_values,error_values_speed_all[:,6],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()

    plt.figure()
    plt.subplot(411)
    plt.plot(az_values,error_values_perc_all[:,3],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values,error_values_perc_all[:,4],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(az_values,error_values_perc_all[:,5],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    plt.plot(az_values,error_values_perc_all[:,6],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    """
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(az_values_cars,error_values_dist_car[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values_cars,error_values_dist_car[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(az_values_cars,error_values_dist_car[:,2],'.')
    plt.xlabel('azimuth values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.title('Sun azimuth and distance errors: Car trajectory 2')
    plt.show()
    """
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(az_values_cars,error_values_speed_car[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values_cars,error_values_speed_car[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(az_values_cars,error_values_speed_car[:,2],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()


    plt.figure()
    plt.subplot(311)
    plt.plot(az_values_bikes,error_values_dist_bike[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values_bikes,error_values_dist_bike[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(az_values_bikes,error_values_dist_bike[:,2],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()

    plt.figure()
    plt.subplot(311)
    plt.plot(az_values_bikes,error_values_speed_bike[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values_bikes,error_values_speed_bike[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(az_values_bikes,error_values_speed_bike[:,2],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    
    plt.figure()
    plt.subplot(311)
    plt.plot(az_values_bikes,error_values_perc_bike[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values_bikes,error_values_perc_bike[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(az_values_bikes,error_values_perc_bike[:,2],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    """
    

    """
    plt.figure()
    plt.subplot(411)
    plt.plot(alt_values,step_values_all[:,3],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(alt_values,step_values_all[:,4],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(alt_values,step_values_all[:,5],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    plt.plot(alt_values,step_values_all[:,6],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    """
        
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(luminance_values[:,0],error_values_dist_car[:,0],'.')
    plt.xlabel('luminance values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(luminance_values[:,1],error_values_dist_car[:,1],'.')
    plt.xlabel('luminance values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(luminance_values[:,2],error_values_dist_car[:,2],'.')
    plt.xlabel('luminance values')
    plt.ylabel('distance errors')
    plt.show()
    """
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(alt_values_cars,error_values_speed_car[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(alt_values_cars,error_values_speed_car[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(alt_values_cars,error_values_speed_car[:,2],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()

    plt.figure()
    plt.subplot(311)
    plt.plot(alt_values_bikes,error_values_speed_bike[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(alt_values_bikes,error_values_speed_bike[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(alt_values_bikes,error_values_speed_bike[:,2],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    """
    """
    ax = plt.axes(projection='3d')
    ax.scatter3D(alt_values_cars, cl_values_cars, error_values_dist_car[:,1])
    plt.show
    """
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(alt_values_cars,error_values_dist_car[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(alt_values_cars,error_values_dist_car[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(alt_values_cars,error_values_dist_car[:,2],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    """
    """
    plt.figure()
    #plt.subplot(411)
    plt.plot(alt_values,error_values_dist_all[:,3],'.')
    plt.xlabel('altitude values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.title('Sun altitude and distance errors: Walker trajectory 0')
    plt.show()
    plt.subplot(312)
    plt.plot(alt_values,error_values_dist_all[:,4],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(alt_values,error_values_dist_all[:,5],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    plt.plot(alt_values,error_values_dist_all[:,6],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.show()
    """
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(az_values_bikes,error_values_perc_bike[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(az_values_bikes,error_values_perc_bike[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    """
    """
    plt.plot(az_values_bikes,error_values_perc_bike[:,2],'.')
    plt.title('Sun azimuth and misclassifications: Bicycle trajectory 2')
    plt.xlabel('azimuth values (degrees)')
    plt.ylabel('distance errors (m)')
    plt.show()
    
    plt.figure()
    plt.subplot(311)
    plt.plot(cl_values_bikes,error_values_perc_bike[:,0],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(cl_values_bikes,error_values_perc_bike[:,1],'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    
    plt.plot(cl_values_bikes,error_values_perc_bike[:,2],'.')
    plt.title('Cloudiness and misclassifications: Bicycle trajectory 2')
    plt.xlabel('cloudiness values')
    plt.ylabel('distance errors (m)')
    plt.show()
    """
    
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(cl_values_cars,error_values_dist_car[:,0],'.')
    plt.xlabel('cloud values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(cl_values_cars,error_values_dist_car[:,1],'.')
    plt.xlabel('cloud values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(cl_values_cars,error_values_dist_car[:,2],'.')
    plt.xlabel('cloud values')
    plt.ylabel('distance errors (m)')
    plt.title('Cloudiness and distance errors: Car trajectory 2')
    plt.show()
    """
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(cl_values,error_values_dist_all[:,3],'.')
    plt.xlabel('cloud values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(cl_values,error_values_dist_all[:,4],'.')
    plt.xlabel('cloud values')
    plt.ylabel('distance errors')
    plt.subplot(313)
    plt.plot(cl_values,error_values_dist_all[:,5],'.')
    plt.xlabel('cloud values')
    plt.ylabel('distance errors (m)')
    plt.title('Cloudiness and distance errors: Car trajectory 2')
    plt.show()
    """
    """
    plt.figure()
    plt.subplot(311)
    plt.plot(alt_values,dist_errors,'.')
    plt.xlabel('altitude values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(alt_values,speed_errors,'.')
    plt.xlabel('altitude values')
    plt.ylabel('speed errors')
    plt.subplot(313)
    plt.plot(alt_values,perc_errors,'.')
    plt.xlabel('altitude values')
    plt.ylabel('perc misclass')
    plt.show()

    plt.figure()
    plt.subplot(311)
    plt.plot(cl_values,dist_errors,'.')
    plt.xlabel('cloud values')
    plt.ylabel('distance errors')
    plt.subplot(312)
    plt.plot(cl_values,speed_errors,'.')
    plt.xlabel('cloud values')
    plt.ylabel('speed errors')
    plt.subplot(313)
    plt.plot(cl_values,perc_errors,'.')
    plt.xlabel('cloud values')
    plt.ylabel('perc misclass')
    plt.show()
    """
    
 
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
