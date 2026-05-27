"""
Defensive half of the capture-team agents.

This file is a focused, standalone reference copy of the DefensiveAgent
defined in `capture.py`. The submission file is still `capture.py`;
this file exists so the defensive logic can be read in isolation.
"""

import typing

import pacai.agents.greedy
import pacai.capture.gamestate
import pacai.core.action
import pacai.core.agent
import pacai.core.board
import pacai.core.features
import pacai.core.gamestate
import pacai.search.distance

SCARED_TIMER_SAFE: int = 5
""" If our scared timer is above this, we treat ourselves as safe to chase invaders again. """

RAID_OPPORTUNITY_RANGE: float = 6.0
""" Max maze distance for an enemy food pellet to count toward a defensive raid opportunity. """

RAID_CLUSTER_THRESHOLD: int = 3
""" Minimum number of nearby pellets required before the defender considers a raid. """

RAID_SAFE_GHOST_RANGE: float = 5.0
""" The nearest non-scared enemy ghost must be farther than this for a raid to be initiated. """

class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes defending its own territory.
    Hunts down invaders, patrols the home boundary and food clusters
    when no invader is visible, and keeps distance when scared.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_defensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        """ Precomputed maze distances. """

        self._home_boundary: list[pacai.core.board.Position] = []
        """ Cells on our side adjacent to the midline. Used for patrolling. """

        # Base feature weights.
        self.weights['on_home_side'] = 200.0
        self.weights['stopped'] = -100.0
        self.weights['reverse'] = -2.0
        self.weights['num_invaders'] = -1000.0
        self.weights['distance_to_invader'] = -25.0
        self.weights['distance_to_boundary'] = -1.0
        self.weights['distance_to_defended_food'] = -1.0
        self.weights['scared_distance'] = 5.0
        self.weights['food_cluster_distance'] = -50.0
        self.weights['enemy_ghost_close'] = -100.0

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
    The defense patrols these cells when no invader is visible.
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

def _extract_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if (current_position is None):
        # We are dead and waiting to respawn.
        return features

    # Strongly prefer being on our own side.
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prefer not turning around.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if (len(agent_actions) > 1):
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    # Count and chase visible invaders.
    invader_positions = state.get_invader_positions(agent_index = agent.agent_index)
    features['num_invaders'] = len(invader_positions)

    if (len(invader_positions) > 0):
        invader_distance = _min_distance(agent._distances, current_position, invader_positions.values())
        if (invader_distance is not None):
            if (state.is_scared(agent.agent_index)):
                # When we are scared, hovering one step away is ideal:
                # we cannot eat them, and stepping onto them costs us a respawn.
                features['scared_distance'] = -abs(invader_distance - 2)
                features['distance_to_invader'] = 0.0
            else:
                features['distance_to_invader'] = invader_distance
    else:
        # No invaders in view — patrol the home boundary so we intercept new ones early.
        if (len(agent._home_boundary) > 0):
            boundary_distance = _min_distance(agent._distances, current_position, agent._home_boundary)
            if (boundary_distance is not None):
                features['distance_to_boundary'] = boundary_distance

        # Also sit near our defending food clusters; protects the densest area.
        team_modifier = _team_modifier_for(agent.agent_index)
        defending_food = state.get_food(team_modifier = -team_modifier)
        if (len(defending_food) > 0):
            food_distance = _min_distance(agent._distances, current_position, defending_food)
            if (food_distance is not None):
                features['distance_to_defended_food'] = food_distance

        # Opportunistic offense: when no invaders to chase, look for nearby clusters of enemy food.
        enemy_food = state.get_food(agent_index = agent.agent_index)
        nearby_food_distances: list[float] = []
        for food_position in enemy_food:
            distance = agent._distances.get_distance(current_position, food_position)
            if ((distance is not None) and (distance <= RAID_OPPORTUNITY_RANGE)):
                nearby_food_distances.append(distance)

        nonscared_enemies = list(state.get_nonscared_opponent_positions(agent_index = agent.agent_index).values())
        ghost_distance = _min_distance(agent._distances, current_position, nonscared_enemies)

        is_raiding = state.is_pacman(agent.agent_index)
        has_cluster = (len(nearby_food_distances) >= RAID_CLUSTER_THRESHOLD)
        is_safe = ((ghost_distance is None) or (ghost_distance > RAID_SAFE_GHOST_RANGE))

        # Initiate a raid on a fresh cluster opportunity, or keep going if already committed.
        if (is_raiding or (has_cluster and is_safe)):
            if (len(nearby_food_distances) > 0):
                features['food_cluster_distance'] = min(nearby_food_distances)
                # Cancel the home-side anchor so the cluster pull can win.
                features['on_home_side'] = 0

        # While exposed in enemy territory, punish proximity to non-scared enemy ghosts.
        if (is_raiding and (ghost_distance is not None) and (ghost_distance <= RAID_SAFE_GHOST_RANGE)):
            features['enemy_ghost_close'] = RAID_SAFE_GHOST_RANGE - ghost_distance

    return features
