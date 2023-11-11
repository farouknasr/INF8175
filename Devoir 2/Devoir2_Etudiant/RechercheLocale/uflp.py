import math
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class UFLP:

    def __init__(self, instance_name: str) -> None:
        self.instance_name = instance_name
        self.load_instance(instance_name)

    def load_instance(self, instance_name: str) -> None:
        self.main_stations_opening_cost = []
        self.main_stations_coordinates = []
        self.satellite_stations_connection_coordinates = []
        self.satellite_stations_connection_cost = []
        filename = f"instances/{instance_name}.txt"
        with open(filename, "r") as f:
            lines = f.readlines()
            self.n_main_station = int(lines[0].split()[0])
            self.n_satellite_station = int(lines[0].split()[1])
            for i in range(1, self.n_main_station + 1):
                line = lines[i].split()
                self.main_stations_opening_cost.append(float(line[2]))
                self.main_stations_coordinates.append((float(line[0]), float(line[1])))
            for i in range(self.n_main_station + 1, self.n_main_station + self.n_satellite_station + 1):
                line = lines[i].split()
                self.satellite_stations_connection_coordinates.append((float(line[0]), float(line[1])))
        self.satellite_stations_connection_cost = [
            [self.coordinates_to_cost(self.main_stations_coordinates[i][0], self.main_stations_coordinates[i][1],
                                      self.satellite_stations_connection_coordinates[j][0], self.satellite_stations_connection_coordinates[j][1])
             for j in range(self.n_satellite_station)] for i in range(self.n_main_station)]

    def coordinates_to_cost(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calcultate_cost(self, main_stations_opened: List[int], satellite_stations_association: List[int]) -> float:
        if sum(main_stations_opened) == 0:
            return math.inf
        opening_cost = sum(main_stations_opened[i] * self.main_stations_opening_cost[i] for i in range(self.n_main_station))
        distance_cost = sum(self.satellite_stations_connection_cost[satellite_stations_association[i]][i] for i in range(self.n_satellite_station))
        return opening_cost + distance_cost

    def get_opening_cost(self, main_stations: int) -> float:
        return self.main_stations_opening_cost[main_stations]

    def get_association_cost(self, main_station: int, satellite_station: int) -> float:
        return self.satellite_stations_connection_cost[main_station][satellite_station]

    def show_solution(self, main_stations_opened: List[int], satellite_stations_association: List[int]) -> None:
        plt.figure(figsize=(10, 7))
        # ... existing plotting code remains unchanged ...

    def solution_checker(self, main_stations_opened: List[int], satellite_stations_association: List[int]) -> bool:
        if len(main_stations_opened) != self.n_main_station:
            print("Wrong solution: length of opened main stations does not match the number of main stations")
            return False
        if len(satellite_stations_association) != self.n_satellite_station:
            print("Wrong solution: length of associated satellite stations does not match the number of satellite stations")
            return False
        if sum(main_stations_opened) == 0:
            print("Wrong solution: no main station opened")
            return False
        for main_station in satellite_stations_association:
            if main_station < 0 or main_station >= self.n_main_station:
                print(f"Wrong solution: index of main station {main_station} is out of bounds")
                return False
            if not main_stations_opened[main_station]:
                print(f"Wrong solution: satellite station assigned to closed main station {main_station}")
                return False
        for state in main_stations_opened:
            if state not in [0, 1]:
                print("Wrong solution: value different than 0/1 in main_stations_opened")
                return False
        return True
