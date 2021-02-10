# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:05:55 2020

@author: stitch
"""
import random
import numpy as np
import os

# Data of the environment
zone = random.randint(3, 6)  #basestation 3~5
zone=3 
#capacity = np.random.randint(35, 40, zone)
capacity=np.array([35,35,35])
#distribution = np.random.randint(60, 90, zone)
distribution = np.array([30,30,30])
bike_num = sum(distribution)


def demand_bound():
    """
    Write demand bound information into file.

    The first line is the number of bikes (bike_num).
    The second line is the capacity of each zone (capacity).
    The third line is the starting allocation of each zone (distribution).
    """
    f = open("demand_bound.txt", "w+")
    f.write(str(zone) + "\n")
    for i in range(0, zone):
        f.write(str(capacity[i]) + ' ')
    f.write(" \n")
    for i in range(0, zone):
        f.write(str(distribution[i]) + ' ')
    f.write("\n")

    f.close()


def distance_zone():
    """
    Write distance zone information into file.
    The data is a matrix with the shape (zone, zone).
    Each element is the distance between two zones.
    For example, element in row i, column j denotes the distance from 
    zone i to zone j.
    """
    dist = np.random.uniform(0.000000001, 12, (zone, zone))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # make it symmetry

    fx = open("distance_zone.txt", "w+")
    for i in range(zone):
        for j in range(zone):
            fx.write(str(dist[i][j]) + ' ')
        fx.write("\n")
    fx.close()


def demand_scenario(scenario_num=500, ntimesteps=100):
    """
    Write demand scenario information into file.

    The data is a 3d array with the shape (ntimesteps, zone, zone)
    Each element is the demand flow between two zones at a certain timestep.
    For example, element in index [i][j][k] denotes the bikes are moved from
    zone j to zone k at timestep i.
    """
    if os.path.isdir("demand_scenario/") == False:
        os.mkdir("demand_scenario/")
    for scenario in range(1, scenario_num + 1):
        file_name = "demand_scenario/demand_scenario_{}".format(scenario)
        f = open(file_name, "w+")
        for i in range(ntimesteps):
            for j in range(zone):
                for k in range(zone):
                    demand = float(np.random.randint(5, 26)) if j != k else 0.0
                    f.write(str(demand) + ' ')
                f.write("\n")
        f.close()


if __name__ == "__main__":
    print("zone:", zone, "capacity:", capacity, "distribution:", distribution,
          "bike,_num", bike_num)
    demand_bound()
    distance_zone()
    demand_scenario()