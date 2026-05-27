"""
Offensive half of the capture-team agents.

This file is a focused, standalone reference copy of the OffensiveAgent
defined in `capture.py`. The submission file is still `capture.py`;
this file exists so the offensive logic can be read in isolation.
"""

import typing

import pacai.agents.greedy
import pacai.capture.gamestate
import pacai.core.action
import pacai.core.agent
import pacai.core.board
import pacai.core.features
import pacai.core.gamestate
import pacai.pacman.board
import pacai.search.distance

GHOST_DANGER_RANGE: float = 5.0
""" Distance at which a non-scared enemy ghost starts pushing us away. """

GHOST_PANIC_RANGE: float = 2.0
""" Distance at which a non-scared enemy ghost triggers a binary panic feature. """

CAPSULE_PRIORITY_RANGE: float = 6.0
""" Distance at which an enemy capsule is treated as a priority pickup. """

class OffensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes eating enemy food.
    Avoids non-scared enemy ghosts, prefers capsules under threat,
    and retreats toward the home boundary when pressured on enemy territory.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_offensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        """ Precomputed maze distances. """

        self._home_boundary: list[pacai.core.board.Position] = []
        """ Cells on our side adjacent to the midline. Used to plan retreats. """

        # Base feature weights.
        self.weights['score'] = 100.0
        self.weights['food_remaining'] = -5.0
        self.weights['distance_to_food'] = -2.0
        self.weights['distance_to_capsule'] = -3.0
        self.weights['ghost_distance'] = 4.0
        self.weights['ghost_panic'] = -500.0
        self.weights['scared_ghost_distance'] = -2.0
        self.weights['distance_to_home'] = -1.0
        self.weights['stopped'] = -100.0
        self.weights['reverse'] = -2.0
        self.weights['dead_end'] = -50.0

        if (override_weights is None):
            override_weights = {}

        for (key, weight) in override_weights.items():
            self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)
        self._home_boundary = _compute_home_boundary(
                typing.cast(pacai.capture.gamestate.GameState, initial_state), self.agent_index)

def _team_modifier_for(agent_index: int) -> int:
    """ Derive team modifier (-1 for the even team, +1 for the odd team). """

    return ((agent_index % 2) * 2) - 1

def _compute_home_boundary(
        state: pacai.capture.gamestate.GameState,
        agent_index: int,
        ) -> list[pacai.core.board.Position]:
    """
    Return non-wall positions on our team's half that lie adjacent to the opposite half.
    The offense retreats toward the nearest one when running for home.
    """

    board = state.board
    team_modifier = _team_modifier_for(agent_index)
    midline_col = board.width // 2

    if (team_modifier > 0):
        boundary_col = midline_col
    else:
        boundary_col = midline_col - 1

    boundary: list[pacai.core.board.Position] = []
    for row in range(board.height):
        position = pacai.core.board.Position(row, boundary_col)
        if (not board.is_wall(position)):
            boundary.append(position)

    return boundary

def _min_distance(
        distances: pacai.search.distance.DistancePreComputer,
        source: pacai.core.board.Position,
        targets: typing.Iterable[pacai.core.board.Position],
        ) -> float | None:
    """ Return the minimum maze distance from source to any target, or None if undefined. """

    best: float | None = None
    for target in targets:
        distance = distances.get_distance(source, target)
        if (distance is None):
            continue

        if ((best is None) or (distance < best)):
            best = distance

    return best

def _get_capsule_positions(
        state: pacai.capture.gamestate.GameState,
        agent_index: int,
        ) -> list[pacai.core.board.Position]:
    """ Return capsule positions on the opponent's side (the ones we can eat). """

    team_modifier = _team_modifier_for(agent_index)
    midline = state.board.width / 2
    capsules: list[pacai.core.board.Position] = []
    for position in state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE):
        position_side = -1 if (position.col < midline) else 1
        if (position_side != team_modifier):
            capsules.append(position)

    return capsules

def _count_open_neighbors(
        state: pacai.capture.gamestate.GameState,
        position: pacai.core.board.Position,
        ) -> int:
    """ Count the number of non-wall neighbors of a position. """

    return len(state.board.get_neighbors(position))

def _extract_offensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(OffensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()
    features['score'] = state.get_normalized_score(agent.agent_index)

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prefer not turning around.
    # The state we receive is already a successor, so the previous action is two back.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if (len(agent_actions) > 1):
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    current_position = state.get_agent_position(agent.agent_index)
    if (current_position is None):
        # We are dead and waiting to respawn. Nothing more to evaluate.
        return features

    # Pressure to clear out enemy food.
    food_positions = state.get_food(agent_index = agent.agent_index)
    features['food_remaining'] = len(food_positions)

    if (len(food_positions) > 0):
        distance_to_food = _min_distance(agent._distances, current_position, food_positions)
        if (distance_to_food is not None):
            features['distance_to_food'] = distance_to_food
    else:
        # All food eaten — game ending bonus.
        features['distance_to_food'] = -1000.0

    # Distance to closest non-scared enemy ghost we can see.
    nonscared_positions = list(state.get_nonscared_opponent_positions(agent_index = agent.agent_index).values())
    ghost_distance: float | None = None
    if (len(nonscared_positions) > 0):
        ghost_distance = _min_distance(agent._distances, current_position, nonscared_positions)

    # Threat features only apply while we are vulnerable on enemy territory.
    if (state.is_pacman(agent.agent_index) and (ghost_distance is not None)
            and (ghost_distance <= GHOST_DANGER_RANGE)):
        features['ghost_distance'] = ghost_distance

        if (ghost_distance <= GHOST_PANIC_RANGE):
            features['ghost_panic'] = 1
            # Stepping into a dead-end while a ghost is right behind us is usually fatal.
            if (_count_open_neighbors(state, current_position) <= 1):
                features['dead_end'] = 1

        # When threatened, give us an escape vector toward our home boundary.
        if (len(agent._home_boundary) > 0):
            distance_to_home = _min_distance(agent._distances, current_position, agent._home_boundary)
            if (distance_to_home is not None):
                features['distance_to_home'] = distance_to_home

    # Scared enemies are bonus food: chase them whenever we can.
    scared_positions = list(state.get_scared_opponent_positions(agent_index = agent.agent_index).values())
    if (len(scared_positions) > 0):
        scared_distance = _min_distance(agent._distances, current_position, scared_positions)
        if (scared_distance is not None):
            features['scared_ghost_distance'] = scared_distance

    # Capsules are most valuable when an enemy ghost is close.
    capsule_positions = _get_capsule_positions(state, agent.agent_index)
    if (len(capsule_positions) > 0):
        capsule_distance = _min_distance(agent._distances, current_position, capsule_positions)
        if (capsule_distance is not None):
            urgency = 1.0
            if ((ghost_distance is not None) and (ghost_distance <= CAPSULE_PRIORITY_RANGE)):
                urgency = 3.0

            features['distance_to_capsule'] = capsule_distance * urgency

    return features
