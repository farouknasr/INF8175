# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from custom_types import Direction
from pacman import GameState
from typing import Any, Tuple,List
import util

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self)->Any:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state:Any)->bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state:Any)->List[Tuple[Any,Direction,int]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions:List[Direction])->int:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def tinyMazeSearch(problem:SearchProblem)->List[Direction]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem:SearchProblem)->List[Direction]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 1 ICI
    '''

    s = problem.getStartState()                # Initialisation du premier élement 's'
    L = util.Stack()                           # Fringe 'L' en structure LIFO (pour DFS)
    L.push((s,[]))                             # Ajout de l'élement 's' à la structure
    V = []                                     # Initialisation des élements visités 'V' (graph search)
    while not L.isEmpty():                     # Tant que 'L' n'est pas vide
        s = L.pop()                            # Retirer de 'L' le dernier élement ajouté
        if s[0] not in V:                      # Si l'élement 's' n'a pas déjà été visité
            if problem.isGoalState(s[0]):      # Si l'élement 's' est celui recherché
                return s[1]                    # Retourne les actions menant à l'élement 's'
            else:                              # Sinon
                C = problem.getSuccessors(s[0])# Générer la liste 'C' de successeurs de l'élement 's'                 
                for c in C:                    # Pour chaque élement 'c' dans la liste 'C'
                    if c[0] not in V:          # Si l'élement 'c' n'a pas déjà été visité
                        a = list(s[1])         # Initialiser la liste d'actions 'a' contenant en premier lieu les actions menant à 's'
                        a.append(c[1])         # Ensuite en rajouter les actions menant à 'c' de 's'
                        L.push((c[0],a))       # Ajouter l'élément 'c' (y compris ses actions) à 'L'
                V.append(s[0])                 # Ajouter 's' à la liste 'V' pour qu'il ne soit plus visité
    util.raiseNotDefined()


def breadthFirstSearch(problem:SearchProblem)->List[Direction]:
    """Search the shallowest nodes in the search tree first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI
    '''
    s = problem.getStartState()                # Initialisation du premier élement 's'
    L = util.Queue()                           # Fringe 'L' en structure FIFO (pour BFS)
    L.push((s,[]))                             # Ajout de l'élement 's' à la structure
    V = []                                     # Initialisation des élements visités 'V' (graph search)
    while not L.isEmpty():                     # Tant que 'L' n'est pas vide
        s = L.pop()                            # Retirer de 'L' le dernier élement ajouté
        if s[0] not in V:                      # Si l'élement 's' n'a pas déjà été visité
            if problem.isGoalState(s[0]):      # Si l'élement 's' est celui recherché
                return s[1]                    # Retourne les actions menant à l'élement 's'
            else:                              # Sinon
                C = problem.getSuccessors(s[0])# Générer la liste 'C' de successeurs de l'élement 's'                 
                for c in C:                    # Pour chaque élement 'c' dans la liste 'C'
                    if c[0] not in V:          # Si l'élement 'c' n'a pas déjà été visité
                        a = list(s[1])         # Initialiser la liste d'actions 'a' contenant en premier lieu les actions menant à 's'
                        a.append(c[1])         # Ensuite en rajouter les actions menant à 'c' de 's'
                        L.push((c[0],a))       # Ajouter l'élément 'c' (y compris ses actions) à 'L'
                V.append(s[0])                 # Ajouter 's' à la liste 'V' pour qu'il ne soit plus visité
    util.raiseNotDefined()

def uniformCostSearch(problem:SearchProblem)->List[Direction]:
    """Search the node of least total cost first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''
    s = problem.getStartState()                # Initialisation du premier élement 's'
    L = util.PriorityQueue()                   # Fringe 'L' en structure coût prioritaire (pour UCS)
    L.push((s,[],0),0)                         # Ajout de l'élement 's' à la structure
    V = []                                     # Initialisation des élements visités 'V' (graph search)
    while not L.isEmpty():                     # Tant que 'L' n'est pas vide
        s = L.pop()                            # Retirer de 'L' le dernier élement ajouté
        if s[0] not in V:                      # Si l'élement 's' n'a pas déjà été visité
            if problem.isGoalState(s[0]):      # Si l'élement 's' est celui recherché
                return s[1]                    # Retourne les actions menant à l'élement 's'
            else:                              # Sinon
                C = problem.getSuccessors(s[0])# Générer la liste 'C' de successeurs de l'élement 's'                 
                for c in C:                    # Pour chaque élement 'c' dans la liste 'C'
                    if c[0] not in V:          # Si l'élement 'c' n'a pas déjà été visité
                        a = list(s[1])         # Initialiser la liste d'actions 'a' contenant en premier lieu les actions menant à 's'
                        a.append(c[1])         # Ensuite en rajouter les actions menant à 'c' de 's'
                        g = float(s[2]+c[2])   # Générer coût total 'g' menant à l'élement 'c'
                        L.update((c[0],a,g),g) # Ajouter l'élément 'c' (y compris ses actions et coûts) à 'L'
                V.append(s[0])                 # Ajouter 's' à la liste 'V' pour qu'il ne soit plus visité
    util.raiseNotDefined()

def nullHeuristic(state:GameState, problem:SearchProblem=None)->List[Direction]:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem:SearchProblem, heuristic=nullHeuristic)->List[Direction]:
    """Search the node that has the lowest combined cost and heuristic first."""
    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 4 ICI
    '''

    s = problem.getStartState()                           # Initialisation du premier élement 's'
    L = util.PriorityQueue()                              # Fringe 'L' en structure coût prioritaire (pour A-star search)
    h = float(heuristic(s,problem))                       # Initialisation de l'heuristique du premier élement 's'
    L.push((s,[],0),h)                                    # Ajout de l'élement 's' à la structure avec en priorité l'heuristique seulement (pas de coût encore)
    V = []                                                # Initialisation des élements visités 'V' (graph search)
    while not L.isEmpty():                                # Tant que 'L' n'est pas vide
        s = L.pop()                                       # Retirer de 'L' le dernier élement ajouté
        if s[0] not in V:                                 # Si l'élement 's' n'a pas déjà été visité
            if problem.isGoalState(s[0]):                 # Si l'élement 's' est celui recherché
                return s[1]                               # Retourne les actions menant à l'élement 's'
            else:                                         # Sinon
                C = problem.getSuccessors(s[0])           # Générer la liste 'C' de successeurs de l'élement 's'                 
                for c in C:                               # Pour chaque élement 'c' dans la liste 'C'
                    if c[0] not in V:                     # Si l'élement 'c' n'a pas déjà été visité
                        a = list(s[1])                    # Initialiser la liste d'actions 'a' contenant en premier lieu les actions menant à 's'
                        a.append(c[1])                    # Ensuite en rajouter les actions menant à 'c' de 's'
                        g = float(s[2]+c[2])              # Générer le coût total 'g' menant à l'élement 'c'
                        h = float(heuristic(c[0],problem))# Déduire l'heuristique de l'élément 'c'
                        f = g + h                         # Calculer la priorité 'f' de l'élément 'c' en sommant son coût avec son heuristique
                        L.update((c[0],a,g),f)            # Ajouter l'élément 'c' (y compris ses actions, coûts et priorité) à 'L'
                V.append(s[0])                            # Ajouter 's' à la liste 'V' pour qu'il ne soit plus visité
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
