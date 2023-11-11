# solver.py
from uflp import UFLP
from typing import List, Tuple
import random

def local_search(problem: UFLP, initial_solution: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
    current_solution = initial_solution
    current_cost = problem.calcultate_cost(*current_solution)
    improved = True

    while improved:
        improved = False
        # Générer des voisins en changeant l'état d'une station principale à la fois
        for i in range(problem.n_main_station):
            neighbor_main_stations = current_solution[0][:]
            neighbor_main_stations[i] = 1 - neighbor_main_stations[i]  # Changer l'état de la station principale

            # Pour chaque station satellite, trouver la station principale ouverte la moins chère à laquelle se connecter
            neighbor_satellite_association = []
            for j in range(problem.n_satellite_station):
                costs = [problem.get_association_cost(main_station, j) if neighbor_main_stations[main_station] == 1 else float('inf') for main_station in range(problem.n_main_station)]
                min_cost_main_station = costs.index(min(costs))
                neighbor_satellite_association.append(min_cost_main_station)

            neighbor_cost = problem.calcultate_cost(neighbor_main_stations, neighbor_satellite_association)
            # Si la solution voisine est meilleure, passer à cette solution
            if neighbor_cost < current_cost:
                current_solution = (neighbor_main_stations, neighbor_satellite_association)
                current_cost = neighbor_cost
                improved = True
                break  # Sortir de la boucle pour recommencer la recherche à partir de la nouvelle solution actuelle

    return current_solution

def solve(problem: UFLP) -> Tuple[List[int], List[int]]:
    # Générer une solution initiale aléatoire
    initial_main_stations = [random.choice([0, 1]) for _ in range(problem.n_main_station)]
    initial_satellite_association = [random.choice(range(problem.n_main_station)) for _ in range(problem.n_satellite_station)]

    # Effectuer une recherche locale à partir de la solution initiale
    solution = local_search(problem, (initial_main_stations, initial_satellite_association))
    return solution