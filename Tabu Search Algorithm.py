# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random
from scipy.optimize import linprog # Imports the library that will enable us to solve the linear programming problem (ie the library that will enable us to implement the simplex solution)
import time                    # Required library to calculate Computational Time

start_time = time.time()                 # Keeps the start time

x_coef_list = [[83.16, 122.22, 82.62, 133.50, 81.81, 65.70, 52.89], # Assigns the x coefficients in the objective function of the mathematical model of the problem to a list
          [97.20, 98.28, 75.48, 147.00, 40.50, 70.20, 110.94],
          [123.84, 99.12, 76.16, 130.00, 79.92, 38.40, 101.48],
          [75.60, 123.48, 42.84, 195.00, 72.90, 88.20, 59.34],
          [139.32, 107.10, 78.54, 121.50, 36.45, 65.70, 99.33],
          [83.88, 92.82, 124.10, 134.50, 75.87, 69.90, 48.59]]

x_coefs = pd.DataFrame(x_coef_list, index=range(1,7), columns=range(1,8)) 
farm_demands= pd.Series([36, 42, 34, 50, 27, 30, 43], index=range(1,8)) 
capacities= pd.Series([80,90, 110, 120, 100, 120], index= range(1,7)) 
y_coefs= pd.Series([220, 240, 260, 275, 240, 230], index= range(1,7))


def neighbour_founder(solution):
    
    neighbour_solutions = [] # The variable created to hold the neighbour solutions of a solution
    
    for i in range(6): # In this loop, neighbour solutions are generated and appended to the neighbour_solutions list
        solution_copy = solution.copy() # Create a copy of the solution given as an argument to the function
    
        if solution_copy[i] == 0:            # The index i of this copy is set to 1 if it is 0, 0 if it is 1 (that is, a neighbour solution is created by closing one facility in the solution if it is already open, opening it if it is closed)
            solution_copy[i] = 1
        elif solution_copy[i] == 1:
            solution_copy[i] = 0
            
        neighbour_solutions.append(solution_copy) # Neighbor solutions created are appended to the neighbor_solutions list
            
    return neighbour_solutions
        
    
def simplexsolver(facilities):
    
    coefs = x_coefs.loc[facilities, :].to_numpy().flatten()
    
    facility_number = len(facilities) 
    
    lessorequal_coef = np.zeros((facility_number,facility_number*7), dtype='float64') 
    
    for i in range(facility_number): 
        j = 7 * i
        lessorequal_coef[i][j : j + 7] = farm_demands 
    
    lessorequal_rhs = capacities[facilities] 
    
    equal_coef = np.zeros((7, facility_number * 7)) 
    
    for i in range(7): 
        equal_coef[i][i::7] = [1] * facility_number 
    
    equal_rhs = [1] * 7 
    
    result = linprog(coefs,
                  A_ub=lessorequal_coef, b_ub=lessorequal_rhs, 
                  A_eq=equal_coef, b_eq=equal_rhs, 
                  method='high')  
    return result 


def better_solution_founder(feasible_set):
    solution_to_return = []
    objective_value_of_solution_to_return = np.Inf
    
    # In this loop, the numbers of open facilities in a solution are found and appended to the facilities list
    for solution in feasible_set:
        if solution != [0,0,0,0,0,0]:
            facilities = []
            for i, j in enumerate(solution):
                if j == 1:
                    facilities.append(i+1)
                    
            # The numbers of the open facilities of that potential solution are sent to the simplexsolver function and the cost that will occur if those facilities are opened (in short, the cost of the potential solution) is calculated
            simplex_result = simplexsolver(facilities)
            objective_value = simplex_result.fun + np.sum(y_coefs[facilities])
            
            # If the potential solution is better than the best solution so far, the solution_to_return and objective_value_of_solution_to_return are updated
            if objective_value < objective_value_of_solution_to_return:
                solution_to_return = solution
                objective_value_of_solution_to_return = objective_value
    return solution_to_return, objective_value_of_solution_to_return

def main(tabu_tanure, tabu_list_length):
    while True:
        current_sol = [] 
        for i in range(6): 
            current_sol.append(random.choice([0,1]))  
        if np.sum(current_sol*capacities) > np.sum(farm_demands): 
            break

    tabu_list = []
    
    best_sol = current_sol 
    best_sol, best_sol_cost = better_solution_founder([best_sol]) 
    print("Initial solution: ", best_sol)
    print("Initial solution cost: ", best_sol_cost)
    
    # Taboo 
    h=0 
    while True:
        h+=1
        print("Iteration:",h)
        neighbour_solutions = neighbour_founder(current_sol)
        [[]]
        
        tabu_indexes_list = [] 
        for tabu in tabu_list: 
            tabu_indexes_list.append(tabu[0]) 
        tabu_indexes_list.sort(reverse = True)
        
        for i in tabu_indexes_list: 
            del neighbour_solutions[i] 
        feasible_set = neighbour_solutions 
        
        feasible_set_copy = feasible_set.copy()
        for solution in feasible_set_copy:
            if np.sum(solution*np.array(capacities)) < np.sum(farm_demands):
                feasible_set.remove(solution)
            
        potential_sol, potential_sol_cost  = better_solution_founder(feasible_set) 
        
        for index, (first, second) in enumerate(zip(potential_sol, current_sol)):
            if first != second:
                index_hold=index
                
        tabu_list.append([index_hold, tabu_tanure]) 
        
        for tabu in tabu_list:
            tabu[1] -= 1
        
        for tabu in tabu_list:
            if tabu[1] == 0:
                tabu_list.remove(tabu)
        
        if len(tabu_list) > tabu_list_length:
            tabu_list.pop(0)
        
        current_sol = potential_sol
        current_sol_cost = potential_sol_cost
        print("Current solution: ", current_sol)
        print("Current solution cost: ", current_sol_cost)
        
        if current_sol_cost < best_sol_cost:
            best_sol = current_sol
            best_sol_cost = current_sol_cost
        
        elif current_sol_cost == best_sol_cost:
            break
        
        print("Best solution: ", best_sol)
        print("Best solution cost: ", best_sol_cost)
        
    print("-----------------------RESULTS ACHIEVED----------------------")
    print("Iterations:", h)
    print("Best solution founded: ", best_sol)
    print("Cost of best solution: ", best_sol_cost)
    comp_time = time.time() - start_time
    print(f"-> Computational Time: {comp_time} seconds")     
    
main(2, 2)
    



















