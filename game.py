##### LIBRARIES
import numpy as np
import copy
from dataclasses import dataclass, field
from typing import Callable, Optional, List

##### CLASSES
@dataclass
class State:
    """
    Represents a state of a Generalized Tic Tac Toe game.

    Args:
        state (np.ndarray): Current state of game
        utility (float): Utility score of the state
        turn (str): Indicates player's turn, 'X' or 'O'
        available_actions (list[tuple[int, int]]): List of tuples (i, j) of available actions for the current state
    """

    state: np.ndarray
    utility: float
    turn: str
    available_actions: list[tuple[int, int]]


class Game:
    """
    Class for Generalized Tic Tac Toe.

    This class implements a generalized version of Tic Tac Toe where:
    - The board is of size n*n.
    - A player wins by placing m consecutive symbols in a row, column, or diagonal.
    - The game state is represented in a numpy array where:
        0 represents an empty tile,
        1 represents the X player (X),
        2 represents the O player (O).
    """

    def __init__(self, n: int, target: int, DEBUG_PRINT=False) -> None:
        """ 
        Initialize game settings and state.
        
        Args:
            n (int): Board size n*n
            target (int): Consecutive m spots in a row to win
            DEBUG_PRINT (bool, optional): Enable debug printing. Defaults to False.
        """

        self.DEBUG_PRINT = DEBUG_PRINT
        self.n = n
        self.m = target
        self.initial_state = State(state=np.zeros((n, n), dtype=int), 
                                   utility=0, 
                                   turn="X",
                                   available_actions=self.generate_actions(np.zeros((n, n), dtype=int)))

    # =============================================================================
    # Core Game Functionality
    # =============================================================================

    def to_move(self, state: State) -> str:
        """ 
        Returns current player's turn

        Args:
            state (State): Current state of game

        Returns: 
            (str): Current player's turn as 'X' or 'O'
        """

        return state.turn


    def is_terminal(self, state: State) -> bool:
        """ 
        Check if current state leads to a win for either agent. 
        
        Args:
            state (State): Current state of game

        Returns:
            (bool): State is a terminal or not
        """

        # For minimax, would be wise to check by changing this to check for the state's utility.

        return (self.check_win(state)) or (len(state.available_actions) == 0)


    def utility(self, state: State, player: str) -> float:
        """
        Returns the utility of a state based on player's turn. 
        
        Args:
            state (State): Current state of game
            player (str): String representation of player's turn, 'X' or 'O'

        Returns:
            (float): Utility of that player's state
        """
        
        if player == 'X':
            utility = state.utility
        else:
            utility = -state.utility

        return utility 

    
    def actions(self, state: State) -> list[tuple[int, int]]:
        """
        Returns all available actions.
        
        Args:
            state (State): Current state of game

        Returns:
            (list[tuple[int, int]]): List of tuples (i, j), which are indices of available actions
        """

        return state.available_actions
    

    def result(self, state: State, move: tuple[int, int]) -> State:
        """
        Update game state with new move. 
        
        Args:
            state (State): Current state of game
            move (tuple[int, int]): New move made by player
        
        Returns:
            (State): State of the game
        """
        
        # If not valid move, return same state
        if move not in state.available_actions:
            if self.DEBUG_PRINT:
                print("Invalid move.")
            return state
        
        # Update game state with new move
        new_state = copy.deepcopy(state)
        player_symbol = 1 if new_state.turn == 'X' else 2
        new_state.state[move] = player_symbol
        new_state.available_actions.remove(move)
        self.switch_turn(new_state)
        return State(state=new_state.state,
                     utility=self.compute_evaluation_score(new_state),
                     turn=new_state.turn,
                     available_actions=new_state.available_actions)


    def switch_turn(self, state: State) -> None:
        """ Switch the current player's turn. """

        state.turn = 'O' if state.turn == 'X' else 'X'

        if self.DEBUG_PRINT:
            print(f"Switched to {state.turn}")

    # =============================================================================
    # Evaluation Function
    # =============================================================================

    def compute_evaluation_score(self, state: State, Eval: Optional[Callable[[State], float]] = None) -> float:
        """
        Compute Eval(s) given an evalulation function 
        
        Args:
            state (State): Current state of the game
            Eval (Optional[Callable[[State], float]]): A callable evaluation function Eval, which takes in state and returns a float, i.e. Eval(s) -> float

        Returns:
            (float): Utility score of state
        """

        if Eval is None:
            Eval = self.eval_check_win

        return Eval(state)
    

    def eval_check_win(self, state: State) -> float:
        """
        Return utility = 1 if game state is a winning terminal state, else utility = 0
        
        Args:
            state (State): Current state of game after move has been placed
        
        Returns:
            (float): Utility score
        """
        
        if self.check_win(state):
            return 1.
        else:
            return 0.

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def generate_actions(self, state: np.ndarray) -> list[tuple[int, int]]:
        """
        Generate possible states given state. 
        
        Args:
            state (np.ndarray): Current state of game

        Returns:
            (list[tuple[int, int]]): List of tuples (i, j), which are indices of available actions
        """

        i, j = np.where(state==0)
        possible_moves = list(zip(i, j))
        return possible_moves
    

    def check_win(self, state: State) -> bool:
        """ 
        Check if the current state is a winning state. 
        Aside: Either I could make this the is_terminal() method or I can find terminal state by returning utility score.
        
        Args:
            state (State): Current state of game

        Returns:
            (bool): If a win has occured
        """

        def check_consecutive(sequence: np.ndarray, target: int) -> bool:
            """
            Helper function: Check if there's a sequence of target length for 1 or 2.
            
            Args: 
                sequence (np.ndarray): Sequence to check
                target (int): Target length of consecutive values

            Returns:
                (bool): If there is a consecutive sequence of target length for 1 or 2.
            """

            count = 0
            last_seen = 0
            for value in sequence:
                if value == last_seen and (value == 1 or value == 2):
                    count += 1
                    if count == target:
                        return True
                else:
                    count = 1
                    last_seen = value
            return False


        curr_state = state.state
        n = self.n
        target = self.m

        # Check rows and columns
        for i in range(n):
            if check_consecutive(curr_state[i, :], target) or check_consecutive(curr_state[:, i], target):
                return True

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            if check_consecutive(np.diagonal(curr_state, offset=d), target) or check_consecutive(np.diagonal(np.fliplr(curr_state), offset=d), target):
                return True

        return False

    # =============================================================================
    # Gameplay Methods
    # =============================================================================

    def play_game(self, game_type: str='PvP', agents: Optional[List[Callable[['Game', State], tuple[int, int]]]] = None, first_player: str='X') -> None:
        """ 
        For debugging, play a local match of Generalized Tic Tac Toe. 

        Args:
            type (str): Type of local match, 'PvP' or 'PvAI' or 'AIvAI'
            agents (Optional[List[Callable[['Game', State], tuple[int, int]]]]): A list of agent functions applicable for 'PvAI' or 'AIvAI'. 
                                                                                 For 'PvAI', the list should contain one agent, and for 'AIvAI', two agents.
        """

        # Check if first_player is valid
        if first_player not in ['X', 'O']:
            raise ValueError("first_player must be 'X' or 'O'.")

        # Setup agents based on the game type
        if game_type == "PvP":
            player_agents = [None, None]  # Both players are human
        elif game_type == "PvAI":
            if not agents or len(agents) != 1:
                raise ValueError("PvAI mode requires exactly one agent.")
            player_agents = [None, agents[0]]  # Human vs. AI
        elif game_type == "AIvAI":
            if not agents or len(agents) != 2:
                raise ValueError("AIvAI mode requires exactly two agents.")
            player_agents = agents  # AI vs. AI

        # Game loop
        state = copy.deepcopy(self.initial_state)
        state.turn = first_player  # Set the first player

        while not self.is_terminal(state):
            curr_agent = player_agents[0] if state.turn == 'X' else player_agents[1]
            print("Current turn:", state.turn)

            if curr_agent:
                # It's an AI's turn
                move = curr_agent(self, state)
                print(f"AI ({state.turn}) chooses move: {move[0]}, {move[1]}")
            else:
                # It's a human's turn
                move_input = input("Enter move as tuple i, j: ").strip().split(',')
                move = (int(move_input[0]), int(move_input[1]))
            
            if move in state.available_actions:
                # Process move
                state = self.result(state, move)
                print("Board state:\n", state.state)

                if self.is_terminal(state):
                    if state.utility == 0: # Draw utility = 0
                        print("Game ended in a draw.")
                    else:
                        self.switch_turn(state) # Hardcode solution to switch character so the print is correct :^ )
                        print(f"Game Over. {state.turn} wins!")
                    print(f"Utility score of terminal state is {state.utility}")
                    return
            else:
                print("Invalid move, please try again.")


    def play_game_API(self) -> None:
        """ Play Generalized Tic Tac Toe against other teams via API. """

        raise NotImplementedError


##### TEST PLAY A GAME
GTTT = Game(n=3, target=3)
GTTT.play_game()