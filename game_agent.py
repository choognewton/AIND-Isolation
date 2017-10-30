"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):  # player context is loser
        return float("-inf")
    if game.is_winner(player):  # player context is winner
        return float("inf")

    player_1 = game.active_player
    player_2 = game.inactive_player

    # move count 1st player
    next_game = None  #  initialize var
    p1_move_count = 0  # initialize var
    # count the total moves available for the
    # for the next two moves in all directions
    p1_move_count = len(game.get_legal_moves(player_1))
    for move in game.get_legal_moves(player_1):
        next_game = game.forecast_move(move)
        p1_move_count += len(next_game.get_legal_moves(player_1))

    # move count 2nd player
    next_game = None  #  initialize var
    p2_move_count = 0  #  initialize var
    # count the total moves available for the
    # for the next two moves in all directions
    p2_move_count = len(game.get_legal_moves(player_2))
    for move in game.get_legal_moves(player_2):
        next_game = game.forecast_move(move)
        p2_move_count += len(next_game.get_legal_moves(player_2))

    return float(p1_move_count -  p2_move_count)  # return player1 - player2

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    player_1 = game.active_player
    player_2 = game.inactive_player

    p1_move_count = 0
    p1_move_count = len(game.get_blank_spaces())

    # move count 2nd player
    next_game = None  # initialize var
    p2_move_count = 0  # initialize var
    # count the total moves available for the
    # for the next two moves in all directions
    p2_move_count = len(game.get_legal_moves(player_2))
    for move in game.get_legal_moves(player_2):
        next_game = game.forecast_move(move)
        p2_move_count += len(next_game.get_legal_moves(player_2))

    return float(p1_move_count - p2_move_count)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    own_score = float(len(game.get_blank_spaces()) - len(game.get_legal_moves(player)))
    opp_score = float(len(game.get_legal_moves(game.get_opponent(player))) * int(len(game.get_blank_spaces())/game.move_count))
    return float(own_score - abs(opp_score))

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=60.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        # start time and initialize the best move so that this function returns something
        # in case the search fails due to timeout
        self.time_left = time_left
        chosen_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass

        return chosen_move

    def checktimer(self):  # moved redundant code to function
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
host
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def Min_Value(game, depth):  # min function - param: game, depth
            self.checktimer()  # check if time_left
            if depth == 0:
                return self.score(game, self)  # assess score of the game object
            v = float("inf")  # lowest negative float
            for move in game.get_legal_moves():  # for each move in legal_moves list
                # set v to minimum value returned from Max_Value func and current val of v
                v = min(v, Max_Value(game.forecast_move(move), depth - 1))
            return v  # return the result of the function

        def Max_Value(game, depth):  # max function - param: game, depth
            self.checktimer()  # check if time_left
            if depth == 0:
                return self.score(game, self)  # assess score of the game object
            v = float("-inf")  # max positive float
            for move in game.get_legal_moves():  # for each move in legal_moves list
                # set v to max value returned from Min_Value func and current val of v
                v = max(v, Min_Value(game.forecast_move(move), depth - 1))
            return v  # return the result of the function

        # body of minimax function
        self.checktimer()  # check if time left
        chosen_move = (-1, -1)
        legal_moves = game.get_legal_moves()  # get possible legal moves
        if len(legal_moves) == 0:  # if no moves
            return chosen_move  # return bad move flag
        # choose best move
        chosen_move = max(legal_moves, key=lambda a: Min_Value(game.forecast_move(a), depth - 1))
        return chosen_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left  # initialize time_left
        chosen_move = (-1, -1)  # initialize variable to 'no legal moves'

        if len(game.get_legal_moves()) > 0:  # if there are legal moves
            legal_moves = game.get_legal_moves()  # store legal moves in var
            chosen_move = legal_moves[0]  # select first move in list
        else:  # no moves
            return chosen_move  # return what you have so far.
        try:
            depth = 1
            while True:
                chosen_move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            pass
        return chosen_move

    def checktimer(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def max_value(game, depth, alpha, beta):  # get max value
            self.checktimer()  # check timer
            legal_moves = game.get_legal_moves()  #  legal moves from player context
            if depth < 1 or len(legal_moves) < 1:  # terminal check
                return self.score(game, self)  # score player

            v = float("-inf")  # initialize var
            for move in legal_moves:  # iterate through moves
                #  return minimum value for move
                v = max(v, min_value(game.forecast_move(move), depth - 1, alpha, beta))
                if v >= beta:  # if score0 > upper val
                    return v
                alpha = max(alpha, v)
            return v  # return  val

        def min_value(game, depth, alpha, beta):  # get min value
            self.checktimer()  # check timer
            legal_moves = game.get_legal_moves()  #  legal moves from opponent context
            if depth < 1 or len(legal_moves) < 1:  # terminal check
                return self.score(game, self)  # score player

            v = float("inf")  # initialize var
            for move in legal_moves:  # iterate through moves
                # return max value for move
                v  = min(v, max_value(game.forecast_move(move), depth - 1, alpha, beta))
                if v <= alpha:  # if v <  val
                    return v
                beta = min(beta, v)
            return v

        # main body of alphabeta funtion
        self.checktimer()

        chosen_move = None
        v = float("inf")
        for move in game.get_legal_moves():
            v = min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if v > alpha:
                alpha = v
                chosen_move = move
            alpha = max(alpha, v)
        return chosen_move
