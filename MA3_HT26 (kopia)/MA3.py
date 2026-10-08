""" MA3.py

Student: Martina Zanetti
Mail: martina.zanetti.5654@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import numpy as np
from numba import njit


# Exc1
def approximate_pi(n):
    # n is the number of points      
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    for _ in range(n):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if x**2 + y**2 <= 1:
            red_x.append(x), red_y.append(y)
        else:
            blue_x.append(x), blue_y.append(y)

    pi = 4 * (len(red_x)) / n
    print(f"The number of points are: {n}")
    print()
    print(f"The approximation of π is: {pi}")
    print()

    plt.scatter(red_x, red_y, color='red',s=1)
    plt.scatter(blue_x, blue_y, color ='blue', s=1)
    plt.savefig(f"appriximation_pi({n}).png")
    plt.close()
    
    return pi



# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points

    # d is the number of dimensions of the sphere
    points = [[random.uniform(-1,1) for _ in range(d)] for _ in range(n)] 

    distance = map(lambda p: sum(x**2 for x in p), points)

    inside_ball = list(filter(lambda dist: dist <= 1, distance))

    #volume like above example, but instead hypercube * points inside / total points

    volume = (2**d) * (len(inside_ball)) / n
    return volume

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points

    # d is the number of dimensions of the sphere 
    volume = m.pi**(d/2) / m.gamma(d/2 + 1)
    return volume

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points

    # d is the number of dimensions of the sphere
    #np is the number of processes

    points = np.random.uniform(-1.0, 1.0, (n,d)) #creates nxd matrix with random values between -1 and 1

    inside_counter = 0
    for i in range(n):
        dist_square = 0.0
        for j in range(d):
            dist_square += points[i,j]**2
        if dist_square <= 1.0:
            inside_counter += 1

    return (2**d) * inside_counter / n


#Exc4: parallel code - parallelize actual computations by splitting data
def _inside_counter(args):
    sub_n, d = args
    count = 0
    for _ in range(sub_n):
        points = [random.uniform(-1,1) for _ in range(d)]
        dist_square = sum(x**2 for x in points)
        if dist_square <= 1:
            count += 1
    return count

def sphere_volume_parallel(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    
    points_per_process = n // np

    args = [(points_per_process, d) for _ in range(np)]

    with future.ProcessPoolExecutor(max_workers= np) as executor:
        results = executor.map(_inside_counter, args)

    total_inside = sum(results)

    volume = (2**d) * total_inside / ( points_per_process * np)

    return volume

    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")
    #1st 4.585345375
    #2nd 4.557488041
    #3rd 4.451308666999999

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume_numba(n, d)
    stop = pc()
    print(f"Exc3: numbra time of {d} and {n}: {stop-start}")
    # 1st 1.2707414999999997
    # 2nd 0.6827309169999998
    # 3rd 0.6818163749999995

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    start2 = pc()
    volume_parallel = sphere_volume_parallel(n, d)
    stop2 = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print(f"What is parallel time? It is {stop2 - start2}")
    #sequential time: 4.63

    
    

if __name__ == '__main__':
	main()
