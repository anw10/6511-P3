##### LIBRARIES
import copy
import numpy as np
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, List
from tic import *

##### CLASSES
@dataclass
class State:
    """
    Represents a state of a Generalized Tic Tac Toe game.

    Args:
        state (np.ndarray): Current state of game
        Score (float): Score of the state, in range Utility(loss, p) <= Eval(s, p) <= Utility(win, p)
        turn (str): Indicates player's turn, 'X' or 'O'
        available_actions (list[tuple[int, int]]): List of tuples (i, j) of available actions for the current state
        DEBUG_PRINT (bool): Debug print
    """

    state: np.ndarray
    score: float
    turn: str
    available_actions: list[tuple[int, int]]
    DEBUG_PRINT: bool = field(default=False)


    def __post_init__(self):
        """ Debug print """
        if self.DEBUG_PRINT:
            print(
                f"State initialized with:\n"
                f"State:\n{self.state}\n"
                f"Score: {self.score}\n"
                f"Turn: {self.turn}\n"
                f"Available Actions: {self.available_actions}\n"
                f"----------------"
            )

    def info(self):
        """ Debug print """
        print(
            f"State initialized with:\n"
            f"State:\n{self.state}\n"
            f"Score: {self.score}\n"
            f"Turn: {self.turn}\n"
            f"Available Actions: {self.available_actions}\n"
            f"----------------"
        )


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

    def __init__(self, n: int, target: int, DEBUG_STATE: tuple[int, int]=None, DEBUG_PRINT: bool=False) -> None:
        """
        Initialize game settings and state.

        Args:
            n (int): Board size n*n
            target (int): Consecutive m spots in a row to win
            DEBUG_STATE (tuple[int, int]): Enable play from preset board state for debugging. Takes in tuple (state: np.ndarray, turn: str).
            DEBUG_PRINT (bool): Enable debug printing. Defaults to False.
        """

        self.DEBUG_PRINT = DEBUG_PRINT
        self.n = n
        self.m = target
        if DEBUG_STATE is not None:
            self.initial_state = State(
                state=DEBUG_STATE,
                score=0,
                turn=None,
                available_actions=self.generate_actions(DEBUG_STATE),
            )
            self.initial_state.score = self.compute_evaluation_score(self.initial_state, self.initial_state.turn)
        else:
            self.initial_state = State(
                state=np.zeros((n, n), dtype=int),
                score=0,
                turn="X",
                available_actions=self.generate_actions(np.zeros((n, n), dtype=int)),
            )
        self.agent_symbol = "X"

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

    def is_cutoff(self, state: State, depth: int) -> bool:
        """
        Check if current state is at depth-ply cut off.

        Pseudocode for Quiescence Search adapted from Wikipedia:

            function quiescence_search(state, depth) is
                if state appears quiet or state is a terminal state or depth = 0 then
                    return estimated value of node
                else
                    (recursively search node children with quiescence_search)
                    return estimated value of children

            function is_cutoff(state, depth) is
                if state is a terminal node then
                    return True
                else if depth = 0 then
                    if state appears quiet then
                        return True
                    else
                        return False, continue search with quiescence_search(state, reasonable_depth_value)

        TODO:
            - 1st iteration: Simple is_cutoff functionality, prone to errors due to approximation of non-terminal states
            - 2nd iteration: Quiescence search--evaluation function should be applied to positions that are quiescent (no more pending moves)

        """

        raise NotImplementedError

    def eval(self, state: State, player: str) -> float:
        """Returns the evaluation score of the current state."""

        raise NotImplementedError

    def is_terminal(self, state: State) -> bool:
        """
        Check if current state is a terminal state (win, loss, or draw).

        Args:
            state (State): Current state of game

        Returns:
            (bool): State is a terminal or not
        """

        # For minimax, would be wise to check by changing this to check for the state's utility.

        return (self.check_win(state)) or (len(state.available_actions) == 0)

    def utility(self, state: State, player: str) -> float:
        """
        Returns the utility of the terminal state based on player's turn.

        Args:
            state (State): Terminal state, e.g. win, loss, or draw.
            player (str): String representation of player's turn, 'X' or 'O'

        Returns:
            (float): Utility of terminal state from player's perspective
        """

        # print(f"in utility, player is {player}, current state score is {state.score}")
        # print(f"agent symbol {self.agent_symbol}")
        if player == self.agent_symbol:
            utility = state.score
        else:
            utility = -state.score
        # print(f"returns utility={utility}")
        # print("----------------")
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
        player_symbol = 1 if new_state.turn == "X" else 2
        new_state.state[move] = player_symbol
        new_state.available_actions.remove(move)
        new_score = self.compute_evaluation_score(new_state, new_state.turn)
        self.switch_turn(new_state)
        return State(
            state=new_state.state,
            score=new_score,
            turn=new_state.turn,
            available_actions=new_state.available_actions,
            DEBUG_PRINT = self.DEBUG_PRINT
        )

    def switch_turn(self, state: State) -> None:
        """Switch the current player's turn."""

        state.turn = "O" if state.turn == "X" else "X"

        # if self.DEBUG_PRINT:
        #     print(f"Switched to {state.turn}")

    # =============================================================================
    # Evaluation Function and Features
    # =============================================================================

    def compute_evaluation_score(
        self, state: State, player: str, Eval: Optional[Callable[[State], float]] = None
    ) -> float:
        """
        Compute Eval(s, p) given an evalulation function

        TODO: Change default score from Util(s, p) to Weighted linear evaluation function Eval(s, p)

        Args:
            state (State): Current state of the game
            player (str): Player who just made the move, 'X' or 'O'
            Eval (Optional[Callable[[State], float]]): A callable evaluation function Eval, which takes in state and returns a float, i.e. Eval(s) -> float

        Returns:
            (float): Score of state
        """

        if Eval is None:
            Eval = self.utility_check_win

        return Eval(state)

    def weighted_linear_evaluation_function(self, state: State, player: str) -> float:
        """
        Weighted linear evaluation function in form = w1*f1 + w2*f2 + ... wn*fn, where wi is weight i and fi is feature i.

        The weights wi should be normalized so that the sum is always within the range of a loss to a win.

        Instrict scale for each feature is [0, 10]. 

        Weights need to be determined using Weighted Majority Algorithm, prolly find a good weight after 800 games or something. Can dynamically adjust weights.
        """

        raise NotImplementedError

    def feature_consecutive_symbols(self, state: State, player: str) -> float:
        """
        This feature returns a score for having a longer consecutive sequence given that there are no opponent symbols blocking the sequence.
        
        TODO: 
            1) One endgame position: FORCED MOVES?
              If you can safely secure m-1 consecutive symbols and you have empty tiles at each end, then it's checkmate. 
                Or, Eval(unblocked m-1) = Util(Win)
              Likewise, having m-2 consecutive symbols will put you in a good position, but opponent would want to block any m-2 attempts.
                Or, Eval(unblocked m-2) < Eval(unblocked m-1)
            2) Maybe it's better if we count the number of unblocked symbols nearby in a sequence,
                so 1011 would be = 3 symbols, which would let us see and plan for potential traps.
               Or, this could be better suited as another feature.

        Args:
            state (State): Current state of game
            player (str): Current player, 'X' or 'O'

        Returns:
            (float): Normalized score = longest length of consecutive symbols / target m
        """

        def check_consecutive_symbol(sequence: np.ndarray, target: int, player_symbol: str) -> int:
            """
            Helper function: Finds the player's longest consecutive symbol in a sequence that doesn't contain opponent symbols.
            
            Args:
                sequence (np.ndarray): Sequence to check
                player (str): Player's consecutive symbol to check

            Returns:
                (int): Number of longest consecutive symbols in sequence
            """

            # In the state array, 1 is X and 2 is O.
            player = 1 if player_symbol == 'X' else 2
            opponent = 2 if player_symbol == 'X' else 1
            
            max_count = 0
            sliding_window = deque(maxlen=target)
            for value in sequence:
                sliding_window.append(value)
                if len(sliding_window) == target:
                    if opponent not in sliding_window:
                        max_count = max(max_count, sliding_window.count(player))

            return max_count

        curr_state = state.state
        n = self.n
        target = self.m
        max_horz = max_vert = max_diag = 0

        # Check rows and columns
        for i in range(n):
            max_horz = max(check_consecutive_symbol(curr_state[i, :], target, player), max_horz)
            max_vert = max(check_consecutive_symbol(curr_state[:, i], target, player), max_vert)

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            max_diag = max(check_consecutive_symbol(np.diagonal(curr_state, offset=d), target, player), max_diag)
            max_diag = max(check_consecutive_symbol(np.diagonal(np.fliplr(curr_state), offset=d), target, player), max_diag)

        # Normalize score
        normalized_score = (max(max_horz, max_vert, max_diag) / target) * 10

        return normalized_score


    def feature_m_minus_1_play(self, state: State, player: str) -> float:
        """
        This feature identifies if the player is one move away from winning by finding an unblocked sequence 
        of m-1 consecutive symbols with open tiles at each end. This situation essentially 
        puts the player in a position where a win is guaranteed on the next move.

        If opponent will not win on next move, perform FORCED MOVE: If play is found, this feature returns score = Inf
        """

        def check_m_minus_1_play(sequence: np.ndarray, target: int, player_symbol: str) -> float:
            """
            Helper function: Find m_minus_1 play
            """

            # In the state array, 1 is X and 2 is O.
            player = 1 if player_symbol == 'X' else 2

            # Window size has to be one larger
            window_size = target + 1

            m_minus_1_play_found = 0
            sliding_window = deque(maxlen=window_size)
            for value in sequence:
                sliding_window.append(value)
                if len(sliding_window) == window_size:
                    if sliding_window[0] == 0 and sliding_window[-1] == 0 and sliding_window.count(player) == target - 1:
                            m_minus_1_play_found = math.inf

            return m_minus_1_play_found

        curr_state = state.state
        n = self.n
        target = self.m
        m1play_horz = m1play_vert = m1play_diag = 0

        # Check rows and columns
        for i in range(n):
            m1play_horz += check_m_minus_1_play(curr_state[i, :], target, player)
            m1play_vert += check_m_minus_1_play(curr_state[:, i], target, player)

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            m1play_diag += check_m_minus_1_play(np.diagonal(curr_state, offset=d), target, player)
            m1play_diag += check_m_minus_1_play(np.diagonal(np.fliplr(curr_state), offset=d), target, player)

        m1play_found = m1play_horz + m1play_vert + m1play_diag

        return m1play_found


    def feature_double_m_minus_2_play(self, state: State, player: str) -> float:
        """
        This features identifies another one move win by finding two unblocked m-2 consecutive symbol sequences.
        This situation puts the player in a position where a win is guaranteed on the next move.

        If opponent will not win on next move, perform FORCED MOVE: If play is found, this feature returns score = Inf
        """

        def check_double_minus_2_play(sequence: np.ndarray, target: int, player_symbol: str) -> int:
            """ 
            Helper function: Check if there are double m-2 plays
            """

            # In the state array, 1 is X and 2 is O.
            player = 1 if player_symbol == 'X' else 2
            opponent = 2 if player_symbol == 'X' else 1

            raise NotImplementedError

        curr_state = state.state
        n = self.n
        target = self.m
        double_m2_play_horz = double_m2_play_vert = double_m2_play_diag = 0

        # Check rows and columns
        for i in range(n):
            double_m2_play_horz += check_double_minus_2_play(curr_state[i, :], target, player)
            double_m2_play_vert += check_double_minus_2_play(curr_state[:, i], target, player)

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            double_m2_play_diag += check_double_minus_2_play(np.diagonal(curr_state, offset=d), target, player)
            double_m2_play_diag += check_double_minus_2_play(np.diagonal(np.fliplr(curr_state), offset=d), target, player)

        double_m2_play = math.inf if (double_m2_play_horz + double_m2_play_vert + double_m2_play_diag) > 1 else 0

        return double_m2_play


    def feature_two_moves_to_potential_win(self, state: State, player: str) -> float:
        """
        This feature identifies unblocked sequences of m-2 consecutive symbols with potential to win
        in two moves. It considers sequences where there's either an empty tile at each end of the 
        sequence or enough space to create a winning sequence of m symbols. This strategy prepares for 
        setting up a win or forcing the opponent to defend, thus creating tactical advantages elsewhere.        
        """

        raise NotImplementedError

    def feature_open_lines(self, state: State, player: str) -> float:
        """
        This feature returns a score for establishing potential/open winning lines.
        Offensive strategy: Expand more winning opportunities.

        Args:
            state (State): Current state of game
            player (str): Current player, 'X' or 'O'

        Returns:
            (int): Count of open lines
        """

        def check_open_line(
            sequence: np.ndarray, target: int, player_symbol: str
        ) -> int:
            """
            Helper function: Check how many open lines there are in the sequence

            Args:
                sequence (np.ndarray): Sequence to check
                target (int): Target length of consecutive values needed to be a winning play
                player (str): Player's open lines

            Returns:
                (int): Number of open lines
            """

            # In the state array, 1 is X and 2 is O.
            player = 1 if player_symbol == "X" else 2

            sliding_window = deque(maxlen=target)
            open_lines = 0
            for value in sequence:
                sliding_window.append(value)
                # Check if window contains only player's symbol and/or empty spaces
                if (sliding_window.count(player) > 0) and (
                    sliding_window.count(0) + sliding_window.count(player) == target
                ):
                    open_lines += 1

            return open_lines

        curr_state = state.state
        n = self.n
        target = self.m
        open_horz = open_vert = open_diag = 0

        # Check rows and columns
        for i in range(n):
            open_horz += check_open_line(curr_state[i, :], target, player)
            open_vert += check_open_line(curr_state[:, i], target, player)

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            open_diag += check_open_line(
                np.diagonal(curr_state, offset=d), target, player
            )
            open_diag += check_open_line(
                np.diagonal(np.fliplr(curr_state), offset=d), target, player
            )

        open_lines = open_horz + open_vert + open_diag

        return open_lines

    def feature_block_opponent(self, state: State, player: str) -> int:
        """
        This feature returns a score for how many open lines you remove from opponent's potential play
        Proactive defensive strategy: Limit opponent options.

        Args:
            state (State): Current state of game
            player (str): Current player, 'X' or 'O'

        Returns:
            (float): Count of blocked opponent lines
        """

        def check_block(sequence: np.ndarray, target: int, player_symbol: str) -> int:
            """
            Helper function: Check how many open lines the player removes from opponent's open lines

            Args:
                sequence (np.ndarray): Sequence to check
                target (int): Target length of consecutive values needed to be a winning play
                player (str): Player's open lines

            Returns:
                (int): Number of blocked opponent lines in sequence
            """

            player = 1 if player_symbol == "X" else 2
            opponent = 2 if player_symbol == "X" else 1

            blocked_lines = 0
            sliding_window = deque(maxlen=target)
            for value in sequence:
                sliding_window.append(value)

                # Conditions to consider a line blocked:
                # 1. The window contains at least one player symbol.
                # 2. The window does not form a complete winning sequence for the opponent.

                if len(sliding_window) == target:
                    if player in sliding_window and opponent in sliding_window:
                        # Count the line as blocked if there's a mix of opponent symbols and player's symbol,
                        # indicating the player has disrupted a potential line for the opponent.
                        blocked_lines += 1

            return blocked_lines

        curr_state = state.state
        n = self.n
        target = self.m
        block_horz = block_vert = block_diag = 0

        # Check rows and columns
        for i in range(n):
            block_horz += check_block(curr_state[i, :], target, player)
            block_vert += check_block(curr_state[:, i], target, player)

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            block_diag += check_block(np.diagonal(curr_state, offset=d), target, player)
            block_diag += check_block(
                np.diagonal(np.fliplr(curr_state), offset=d), target, player
            )

        blocked_opponent_lines = block_horz + block_vert + block_diag

        return blocked_opponent_lines

    def feature_block_imminent_lost(self, state: State, player: str) -> int:
        """
        This feature returns a score for how many imminent losts the player blocks.
        Late game defensive strategy: Block an imminent lost from a trap set by opponent.

        Args:
            state (State): Current state of game
            player (str): Current player, 'X' or 'O'

        Returns:
            (float): Count of blocked imminent losts.
        """

        def check_block_imminent_lost(
            sequence: np.ndarray, target: int, player_symbol: str
        ) -> int:
            """
            Helper function: Check how many imminent losts the player blocks

            Args:
                sequence (np.ndarray): Sequence to check
                target (int): Target length of consecutive values needed to be a winning play
                player (str): Player's symbol, 'X' or 'O'
            
            Returns:
                (int): Number of blocked opponent lines in sequence
            """

            """ 
            NEEDS WORK... Only scores higher for blocks of imminent threats, and score persists, so I'm not sure if that is good.
            Still buggy, it returns an unexpected value in this case
                e.g, For n = 5, m = 4
                     0 0 0 0 0
                     0 0 0 1 0
                     0 0 1 2 0
                     0 2 0 0 0
                     0 0 0 0 0
                Here, player 2 must block the diagonal at (3,1) because it is one step away from an unblocked m-1 play. It returns blocked_lines = 2
                    when in reality it's just one blocked imminent lost play. I guess it technically blocks 2 options (opp playing at either end), but
                    I feel like there is something I need to look at more critically here.
            """

            player = 1 if player_symbol == 'X' else 2
            opponent = 2 if player_symbol == 'X' else 1

            sliding_window = deque(maxlen=target)
            blocked_lines = 0
            for value in sequence:
                sliding_window.append(value)
                # A block occurs if the window has exactly target-1 opponent symbols and 1 empty space.
                # Favors blocks with imminent threat of losing.
                if len(sliding_window) == target:
                    if sliding_window.count(opponent) >= target - 2 and sliding_window.count(player) >= 1:
                        blocked_lines += 1
            
            return blocked_lines

        curr_state = state.state
        n = self.n
        target = self.m
        imminent_lost_horz = imminent_lost_vert = imminent_lost_diag = 0

        # Check rows and columns
        for i in range(n):
            imminent_lost_horz += check_block_imminent_lost(
                curr_state[i, :], target, player
            )
            imminent_lost_vert += check_block_imminent_lost(
                curr_state[:, i], target, player
            )

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            imminent_lost_diag += check_block_imminent_lost(
                np.diagonal(curr_state, offset=d), target, player
            )
            imminent_lost_diag += check_block_imminent_lost(
                np.diagonal(np.fliplr(curr_state), offset=d), target, player
            )

        blocked_imminent_lost = (
            imminent_lost_horz + imminent_lost_vert + imminent_lost_diag
        )

        return blocked_imminent_lost

    def feature_center_control(self, state: State, player: str) -> int:
        """
        This feature returns a score for taking control of center tiles.
        """

        raise NotImplementedError

    def feature_corner_control(self, state: State, player: str) -> int:
        """
        This feature returns a score for taking control of corner tiles.
        """

        raise NotImplementedError

    def utility_check_win(self, state: State) -> float:
        """
        Return utility = 1 if game state is a winning terminal state, else utility = 0

        Args:
            state (State): Current state of game after move has been placed

        Returns:
            (float): Utility score
        """

        if self.check_win(state):
            return 1.0
        else:
            return 0.0

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

        i, j = np.where(state == 0)
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
            if check_consecutive(curr_state[i, :], target) or check_consecutive(
                curr_state[:, i], target
            ):
                return True

        # Check diagonals
        for d in range(-n + target, n - target + 1):
            if check_consecutive(
                np.diagonal(curr_state, offset=d), target
            ) or check_consecutive(
                np.diagonal(np.fliplr(curr_state), offset=d), target
            ):
                return True

        return False

    # =============================================================================
    # Gameplay Methods
    # =============================================================================

    def play_game(
        self,
        game_type: str = "PvP",
        agents: Optional[List[Callable[["Game", State], tuple[int, int]]]] = None,
        first_player: str = "X",
    ) -> None:
        """
        For debugging, play a local match of Generalized Tic Tac Toe.

        Args:
            type (str): Type of local match, 'PvP' or 'PvAI' or 'AIvAI'
            agents (Optional[List[Callable[['Game', State], tuple[int, int]]]]): A list of agent functions applicable for 'PvAI' or 'AIvAI'.
                                                                                 For 'PvAI', the list should contain one agent, and for 'AIvAI', two agents.
        """

        # Check if first_player is valid
        if first_player not in ["X", "O"]:
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
        state.turn = first_player  # Set the first player in state
        # self.agent_symbol = 'X' if first_player == 'O' else 'O'

        while not self.is_terminal(state):
            curr_agent = player_agents[0] if state.turn == "X" else player_agents[1]
            self.agent_symbol = 'X' if state.turn == 'O' else 'O'
            print("Current turn:", state.turn)

            if curr_agent:
                # It's an AI's turn
                move = curr_agent[0](self, state)
                print(f"AI ({state.turn}) chooses move: {move[0]}, {move[1]}")
            else:
                # It's a human's turn
                move_input = input("Enter move as tuple i, j: ").strip().split(",")
                try:
                    move = (int(move_input[0]), int(move_input[1]))
                except ValueError:
                    move = None

            if move in state.available_actions:
                # Process move
                state = self.result(state, move)
                print("Board state:\n", state.state)

                if self.is_terminal(state):
                    if state.score == 0:  # Draw utility = 0
                        print("Game ended in a draw.")
                    else:
                        self.switch_turn(
                            state
                        )  # Hardcode solution to switch character so the print is correct :^ )
                        print(f"Game Over. {state.turn} wins!")
                    print(f"Utility score of terminal state is {state.score}")
                    print(
                        f"Utility score from human's perspective: {self.utility(state, state.turn)}"
                    )
                    return
            else:
                print("Invalid move, please try again.")

    def play_game_API(self, agent) -> None:
        """
        Play Generalized Tic Tac Toe against other teams via API.
        First, we need to find out if we're going first or second and what agent symbol we are ('X' or 'O')
        """
        
        #TODO: Make sure to delete the api-keys
        x_api_key = "4e96ce62c8512883a2ac"  # Your API-KEY
        user_id = "1210"  # Your ID
        teamid = "1397"   # Your Team ID
        # teamid2 = "1416"  # Enemy Team ID, 5G_UWB
        # gameid = "4751"   # game ID you are playing
        gameid = "4782"
        
        last_movement_info = get_moves(x_api_key, user_id, gameid, count_most_recent_moves="1")
        last_movement_teamid = last_movement_info[0]["teamId"]
        last_movement_symbol = last_movement_info[0]["symbol"]
        print("DEBUG:", last_movement_info)
        print("DEBUG2:", last_movement_teamid, last_movement_symbol)
        
        current_symbol = self.switch_turn_symbols(last_movement_symbol) if last_movement_teamid == teamid else last_movement_symbol
        
        current_state = np.zeros((self.n, self.n), dtype=int)
        print(current_state)
        # print("디버그", get_game_details(x_api_key, user_id, gameid))
        
        current_board_info = get_board_map(x_api_key, user_id, gameid)
        print(current_board_info)
        for index, symbol in current_board_info.items():
            move_index = index.strip().split(",")
            move = (int(move_index[0]), int(move_index[1]))
            current_state[move] = 1 if symbol == "X" else 2
        print(current_state)
        
        state_object = State(state=current_state, score=0, turn=current_symbol, available_actions=self.generate_actions(state=current_state))
    
        # Game loop
        state = copy.deepcopy(state_object)

        while not self.is_terminal(state):
            curr_agent = agent
            self.agent_symbol = 'X' if state.turn == 'O' else 'O'
            print("Current turn:", state.turn)

            # It's an AI's turn
            move = curr_agent[0](self, state)
            print(f"AI ({state.turn}) chooses move: {move[0]}, {move[1]}")

            if move in state.available_actions:
                # Process move
                make_move(x_api_key, user_id, gameid, teamid, where_to_move=move)
            else:
                print("Invalid move, please try again.")
    
    def switch_turn_symbols(self, symbol: str) -> str:
        return "X" if symbol == "O" else "O"


##### TEST PLAY A GAME
# GTTT = Game(n=5, target=4)
GTTT = Game(n=10, target=3)
# GTTT.play_game()
GTTT.play_game_API(agent=minimax)

# Sample states to test features
# Sample 1: A basic winning condition for X with 4 in a row horizontally
sample_1 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0],
    ]
)
state_1 = State(state=sample_1, score=0, turn="X", available_actions=[])

# Sample 2: A diagonal winning condition for O
sample_2 = np.array(
    [
        [2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ]
)
state_2 = State(state=sample_2, score=0, turn="O", available_actions=[])

# Sample 3: A vertical winning condition for X and a blocked condition for O
sample_3 = np.array(
    [
        [1, 0, 0, 0, 2],
        [1, 0, 0, 0, 2],
        [1, 0, 0, 0, 2],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
state_3 = State(state=sample_3, score=0, turn="X", available_actions=[])

# Sample 4: Mixed conditions with some blocking
sample_4 = np.array(
    [
        [1, 1, 0, 2, 2],
        [2, 1, 1, 1, 0],
        [0, 2, 2, 0, 0],
        [1, 1, 0, 1, 0],
        [2, 0, 0, 0, 0],
    ]
)
state_4 = State(state=sample_4, score=0, turn="X", available_actions=[])

# Sample 5: Blocked imminent lost
sample_5 = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 2, 0],
    [0, 2, 0, 0, 2],
    [0, 0, 0, 0, 2]
])
state_5 = State(state=sample_5, score=0, turn='X', available_actions=[])

# print(GTTT.feature_block_imminent_lost(state=state_5, player='O'))
