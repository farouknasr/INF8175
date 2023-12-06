from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState

class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float = 60*15, *args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name, time_limit, *args)
        self.depth_limit = 3                                                                    # Limite de profondeur pour algorithme Minimax

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        best_score = float('-inf')                                                              # Initialisation du meilleure score à l'infini négatif (indéterminé)
        best_action = None                                                                      # Initialisation de la meilleure action (aucune)

        for action in current_state.generate_possible_actions():                                # Itérer chacune des actions possibles générées à l'état de jeu actuel
            """
            Calcul du nouveau score en utilisant l'algorithme Minimax avec alpha-beta pruning.
            Les paramètres alpha et beta sont indéterminés au début (-inf et inf resepctivement).
            L'algorithme roule en premier lieu de la perspective du joueur MAX avec une profondeur initiale de 1.
            """
            value = self.minimax(action.get_next_game_state(), 1, float('-inf'), float('inf'), True)
            if value > best_score:
                best_score = value
                best_action = action

        return best_action                                                                      # Retourner la meilleure action à prendre en fonction du meilleur score obtenu

    def minimax(self, state, depth, alpha, beta, is_maximizing_player):
        if depth == self.depth_limit or state.is_done():                                        # Évaluer si la limite de profondeur a été atteinte
            return self.evaluate(state)                                                         # Si c'est le cas, retourner l'évaluation de l'état de jeu (fct. heurisitique)
        
        if is_maximizing_player:                                                                # Commencer la boucle de la perspective du joueur MAX
            max_eval = float('-inf')                                                            # Initialiser la valeur maximale (ind.)
            for action in state.generate_possible_actions():
                """
                Déterminer la valeur d'alpha en appelant l'algorithme Minimax de la persepctive du joueur MIN.
                Employer la méthode alpha-beta pruning pour pruner les branches non-nécessaires et réduire l'espace de recherche.
                """
                eval = self.minimax(action.get_next_game_state(), depth + 1, alpha, beta, False) # Augmenter la profondeur de 1
                max_eval = max(max_eval, eval)                                                   # Déduire la valeur maximale
                alpha = max(alpha, eval)                                                         # Mettre à jour alpha (si possible)
                if beta <= alpha:                                                                # Si alpha n'est pas strictement inférieur à beta
                    break                                                                        # Pruner la branche 
            return max_eval                                                                      # Retoutner la valeur maximale obtenue
        else:                                                                                    # Évaluation de l'état de jeu de la persepctive du joueur MIN
            min_eval = float('inf')                                                              # Initialiser la valeur minimale (ind.)
            for action in state.generate_possible_actions():
                """
                Déterminer la valeur de beta en appelant l'algorithme Minimax de la persepctive du joueur MAX (avec alpha-beta pruning).
                """ 
                eval = self.minimax(action.get_next_game_state(), depth + 1, alpha, beta, True) # Augmenter la profondeur de 1
                min_eval = min(min_eval, eval)                                                  # Déduire la valeur minimale
                beta = min(beta, eval)                                                          # Mettre à jour beta (si possible)                                             
                if beta <= alpha:                                                               # Si alpha n'est pas strictement inférieur à beta
                    break                                                                       # Pruner la branche 
            return min_eval                                                                     # Retoutner la valeur maximale obtenue

    def evaluate(self, state):
        """
        Fonction heuristique simple implémentée pour évaluer l'état de jeu quand la limite de profondeur est atteinte
        """
        my_marbles = sum(1 for piece in state.get_rep().get_env().values() if piece.get_owner_id() == self.get_id())       # Compter le nb. de billes restantes du joueur 
        opponent_marbles = sum(1 for piece in state.get_rep().get_env().values() if piece.get_owner_id() != self.get_id()) # Compter le nb. de billes restantes de l'adversaire
        """
        Retourner la différence entre le nombre de billes restanta à chaque joueur (positif : avantage pour le joueur, négatif : avantage pour l'adversaire)
        """
        return my_marbles - opponent_marbles
    