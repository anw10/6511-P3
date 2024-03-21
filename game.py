##### LIBRARIES
import numpy as np

##### CLASSES
class Game():
    """
    Class for Generalized Tic Tac Toe.

    This class implements a generalized version of Tic Tac Toe where:
    - The board is of size n*n.
    - A player wins by placing m consecutive symbols in a row, column, or diagonal.
    - The game state is represented in a numpy array where:
        0 represents an empty tile,
        1 represents the first player (X),
        2 represents the second player (O).
    """

    def __init__(self, n: int, target: int, DEBUG_PRINT=False):
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
        self.initial_state = np.zeros((n, n), dtype=int)
        self.available_actions = self.generate_actions(self.initial_state)
        self.turn = 'X'

    # =============================================================================
    # Core Game Functionality
    # =============================================================================

    def to_move(self, state: np.ndarray) -> str:
        """ 
        Returns current player's turn

        Args:
            state (np.ndarray): Current state of game

        Returns: 
            (str): Current player's turn as 'X' or 'O'
        """

        return self.turn


    def is_terminal(self, state: np.ndarray) -> bool:
        """ 
        Check if current state leads to a win for either agent. 
        
        Args:
            state (np.ndarray): Current state of game

        Returns:
            (bool): State is a terminal or not
        """

        return self.check_win(state)


    def utility(self, state: np.ndarray, move: tuple[int, int]):
        """ Returns the utility of a state after making a move. """
        raise NotImplementedError

    
    def actions(self, state: np.ndarray) -> list[tuple[int, int]]:
        """
        Returns all available actions.
        
        Args:
            state (np.ndarray): Current state of game

        Returns:
            (list[tuple[int, int]]): List of tuples (i, j), which are indices of available actions
        """

        return self.available_actions
    

    def result(self, state: np.ndarray, move: tuple[int, int]) -> np.ndarray:
        """
        Update game state with new move. 
        
        Args:
            state (np.ndarray): Current state of game
            move (tuple[int, int]): New move made by player
        
        Returns:
            (np.ndarray): State of the game
        """
        
        # If not valid move, return same state
        if move not in self.available_actions:
            if self.DEBUG_PRINT:
                print("Invalid move.")
            return state
        
        # Update game state with new move
        player_symbol = 1 if self.turn == 'X' else 2
        state[move] = player_symbol
        self.available_actions.remove(move)
        self.switch_turn()
        return state


    def switch_turn(self):
        """ Switch the current player's turn. """

        self.turn = 'O' if self.turn == 'X' else 'X'

        if self.DEBUG_PRINT:
            print(f"Switched to {self.turn}")

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
    

    def check_win(self, state: np.ndarray) -> bool:
        """ 
        Check if the current state is a winning state. 
        Aside: Either I could make this the is_terminal() method or I can find terminal state by returning utility score.
        
        Args:
            state (np.ndarray): Current state of game

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


        n = self.n
        target = self.m

        # Check rows and columns
        for i in range(n):
            if check_consecutive(state[i, :], target) or check_consecutive(state[:, i], target):
                return True

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            if check_consecutive(np.diagonal(state, offset=d), target) or check_consecutive(np.diagonal(np.fliplr(state), offset=d), target):
                return True

        return False

    # =============================================================================
    # Gameplay Methods
    # =============================================================================

    def play_game(self):
        """ 
        For debugging, play a local match of Generalized Tic Tac Toe. 
        We can only play by manually entering in moves. There is no AI as of now.

        TODO: Allow players to be agents (e.g. minimax)
        """

        state = self.initial_state.copy()

        while not self.is_terminal(state):
            print("Current turn:", self.turn)
            move_input = input("Enter move as tuple i, j: ").strip().split(',')
            move = (int(move_input[0]), int(move_input[1]))
            
            if move in self.available_actions:
                state = self.result(state, move)
                print("Board state:\n", state)
                if self.is_terminal(state):
                    self.switch_turn() # Hardcode solution to switch character so the print is correct :^ )
                    print(f"Game Over. {self.turn} wins!")
                    return
            else:
                print("Invalid move, please try again.")

        print("Game ended in a draw.")


    def play_game_API(self):
        """ Play Generalized Tic Tac Toe against other teams via API. """

        raise NotImplementedError


##### TEST PLAY A GAME
GTTT = Game(n=4, target=3)
GTTT.play_game()