import array
import random
import numpy as np
import os
import re
import seaborn
import matplotlib.pyplot as plt

import csv

from simulation_methods import compute_errors

def main():
    """
    This code is for computing the x/y-errors in the estimations.
    It is not relevant to the generation and running of simulated scenarios.
    It is quite unstructured and has few comments, but it is still included for reference.
    """
    op_path = 'C:/Users/elias/outputs_viscando' #Path to OTUS3D estimations
    gt_path = 'C:/Users/elias/ground_truth' #Path to ground truth
    
    actor = 'all'
    x_y_error = 0 #0 is error in x, 1 is error in y
    
    if actor == 'car':
        min_batch = 0
        max_batch = 2
        min_traj = 0
        max_traj = 3

    elif actor == 'bike':
        min_batch = 2
        max_batch = 4
        min_traj = 0
        max_traj = 3

    elif actor == 'walker':
        min_batch = 0
        max_batch = 4
        min_traj = 3
        max_traj = 7

    elif actor == 'all':
        min_batch = 0
        max_batch = 4
        min_traj = 0
        max_traj = 7

    else:
        min_batch = 0
        max_batch = 2
        min_traj = 2
        max_traj = 3
        
    nbr_points_x = 18+19+2
    nbr_points_y = 18+15+2

    groupings_error = np.zeros((nbr_points_x,nbr_points_y))
    groupings_speed = np.zeros((nbr_points_x,nbr_points_y))
    groupings_class = np.zeros((nbr_points_x,nbr_points_y))
    groupings_ctr = np.zeros((nbr_points_x,nbr_points_y))
    groupings_error_abs = np.zeros((nbr_points_x,nbr_points_y))
    groupings_dist = np.zeros((nbr_points_x,nbr_points_y))

    
    for current_gen in range(1,4):
        for batch in range(min_batch, max_batch):
            (avg_errors, dist_results, speed_results, perc_misclass, step_perc, coord_diff, speed_diff, class_diff, coords_gt_save) = compute_errors(batch, 7, 20, current_gen, True, op_path, gt_path)
            for ind_idx in range(20):
                for traj_idx in range(min_traj, max_traj):
                    for coord_idx in range(len(coords_gt_save[ind_idx][traj_idx][0])):
                        x_coord = coords_gt_save[ind_idx][traj_idx][0][coord_idx]
                        y_coord = coords_gt_save[ind_idx][traj_idx][1][coord_idx]
                        if x_coord < -18:
                            x_idx = 0
                        elif x_coord > 19:
                            x_idx = nbr_points_x - 1 #38
                        else:
                            x_idx = np.floor(x_coord) + 19

                        if y_coord < -18:
                            y_idx = 0
                        elif y_coord > 15:
                            y_idx = nbr_points_y - 1 #34
                        else:
                            y_idx = np.floor(y_coord) + 19


                        groupings_error[int(x_idx)][int(y_idx)] += coord_diff[ind_idx][traj_idx][x_y_error][coord_idx]
                        groupings_speed[int(x_idx)][int(y_idx)] += speed_diff[ind_idx][traj_idx][coord_idx]
                        groupings_class[int(x_idx)][int(y_idx)] += class_diff[ind_idx][traj_idx][coord_idx]
                        groupings_error_abs[int(x_idx)][int(y_idx)] += np.abs(coord_diff[ind_idx][traj_idx][x_y_error][coord_idx])
                        groupings_dist[int(x_idx)][int(y_idx)] += np.sqrt(coord_diff[ind_idx][traj_idx][0][coord_idx]**2 + coord_diff[ind_idx][traj_idx][1][coord_idx]**2)
                        groupings_ctr[int(x_idx)][int(y_idx)] += 1
                        
                        #if groupings_error_abs[int(x_idx)][int(y_idx)] > 10:
                        #    print(batch)
                        #    print(ind_idx)
                        
                    #coord = coords_gt_save[ind_idx][traj]
                    #print(coords_gt_save[ind_idx][traj_idx][0][0])
    
    
    #groupings_avgs = groupings_error[groupings_ctr != 0]/groupings_ctr[groupings_ctr != 0]
    #speed_avgs = groupings_speed[groupings_ctr != 0]/groupings_ctr[groupings_ctr != 0]
    #class_avgs = groupings_class[groupings_ctr != 0]/groupings_ctr[groupings_ctr != 0]

    groupings_avgs = groupings_error/groupings_ctr
    speed_avgs = groupings_speed/groupings_ctr
    class_avgs = groupings_class/groupings_ctr
    groupings_avgs_abs = groupings_error_abs/groupings_ctr
    dist_avgs = groupings_dist/groupings_ctr
    

    
    """
    print('-----')
    print(groupings_ctr)
    print(groupings_avgs)
    print(speed_avgs)
    print(class_avgs)
    """

    
    seaborn.heatmap(groupings_avgs)#,vmin=-5)
    plt.ylim(0,nbr_points_x)
    plt.title("Distance errors in the y-direction")
    plt.xlabel('y-coordinate (m)')
    plt.ylabel('x-coordinate (m)')
    plt.arrow(19.5,0,3*np.sin(np.radians(7)),3*np.cos(np.radians(7)), width = 0.003, head_width=300*0.003, head_length=150*0.003)
    plt.show()
    
    #seaborn.heatmap(groupings_avgs_abs)#,vmax=0.6)
    #plt.ylim(0,nbr_points_x)
    #plt.arrow(19.5,0,3*np.sin(np.radians(7)),3*np.cos(np.radians(7)), width = 0.003, head_width=300*0.003, head_length=150*0.003)
    #plt.show()
    """
    seaborn.heatmap(dist_avgs,vmax=5)#,vmax=1.5)#,vmax=3)
    plt.ylim(0,nbr_points_x)
    plt.title("Walker trajectories: Distance errors")
    plt.xlabel('y-coordinate (m)')
    plt.ylabel('x-coordinate (m)')
    plt.arrow(19.5,0,3*np.sin(np.radians(7)),3*np.cos(np.radians(7)), width = 0.003, head_width=300*0.003, head_length=150*0.003)
    plt.show()
    seaborn.heatmap(speed_avgs,vmax=1.7)#,vmax=3)
    plt.ylim(0,nbr_points_x)
    plt.title("Walker trajectories: Speed errors")
    plt.xlabel('y-coordinate (m)')
    plt.ylabel('x-coordinate (m)')
    plt.arrow(19.5,0,3*np.sin(np.radians(7)),3*np.cos(np.radians(7)), width = 0.003, head_width=300*0.003, head_length=150*0.003)
    plt.show()
    seaborn.heatmap(class_avgs)#,vmax=0.8)#,vmax=0.5)
    plt.arrow(19.5,0,3*np.sin(np.radians(7)),3*np.cos(np.radians(7)), width = 0.003, head_width=300*0.003, head_length=150*0.003)
    plt.ylim(0,nbr_points_x)
    plt.title("Walker trajectories: Misclassifications")
    plt.xlabel('y-coordinate (m)')
    plt.ylabel('x-coordinate (m)')
    #ax4.remove()
    #ax6.remove()
    #fig.tight_layout()
    plt.show()
    """

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
