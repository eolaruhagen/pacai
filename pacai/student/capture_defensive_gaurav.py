import pacai.core.agentinfo
import pacai.util.alias
import pacai.agents.greedy
import typing
import pacai.search.distance
import pacai.core.gamestate
import pacai.core.features
import pacai.core.action
import pacai.core.agent
import pacai.core.board
import pacai.capture.gamestate
import pacai.pacman.board

SAFE_INVADER_DISTANCE = 8

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    agent1_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)
    agent2_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)

    return [agent1_info, agent2_info]



# first start with two basic offense defense greedy based agents using the distance precomputer


class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    def __init__(self, **kwargs: typing.Any):
        # init the feature extractor func via the kwargs using my modified feature extraction agents
        kwargs['feature_extractor_func'] = extract_defensive_features
        super().__init__(**kwargs)
        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        """ Precompute distances. """


        # Set base weights.
        self.weights['distance_to_invader'] = -20.0
        self.weights['food_distance_sum'] = -0.1
        self.weights['num_invaders_on_same_side'] = -1000.0
        self.weights['invader_distance_to_food'] = 5.0
       


    def game_start(self, initial_state: pacai.core.gamestate.GameState):
        self._distances.compute(initial_state.board)
        

def extract_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **_kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if (current_position is None):
        # We are dead and waiting to respawn.
        return features


    # Note the side of the board we are on.
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    
    # important features for defense
    # 1. distance_to_invader
    #    If not scared, move toward the closest invader.
    #    If scared, move away from the closest invader, but stop caring after 8 tiles.
    # 2. food_distance_sum
    #    Sum of all distances from us to all food we are defending.
    # 3. num_invaders_on_same_side
    #    Number of invaders on our side of the board.
    # 4. invader_distance_to_food
    #    The closest distance from any invader to any food we are defending.

    invader_positions = state.get_invader_positions(agent_index = agent.agent_index)
    defended_food = get_defended_food(state, agent.agent_index)
    is_scared = state.is_scared(agent_index = agent.agent_index)
# 
    # Feature 3: number of invaders on our side.
    features['num_invaders_on_same_side'] = len(invader_positions)

    # Feature 1: distance to the closest invader.
    # If scared, this becomes a danger score that stops growing once we are far enough away.
    if (len(invader_positions) > 0):
        distances_to_invaders = []

        # Get the distance to each invader
        for invader_position in invader_positions.values():
            distance = agent._distances.get_distance(current_position, invader_position)
            if (distance is not None):
                distances_to_invaders.append(distance)
        # apply feature to closest invader
        if (len(distances_to_invaders) > 0):
            closest_invader = min(distances_to_invaders)

            if (is_scared):
                # 0 means far enough. Bigger means too close and dangerous.
                features['distance_to_invader'] = SAFE_INVADER_DISTANCE - min(
                        closest_invader,
                        SAFE_INVADER_DISTANCE)
            else:
                features['distance_to_invader'] = closest_invader

    # Feature 2: sum of distances from us to the food we are defending.
    food_distance_sum = 0
    for food_position in defended_food:
        distance = agent._distances.get_distance(current_position, food_position)
        if (distance is not None):
            food_distance_sum += distance

    features['food_distance_sum'] = food_distance_sum

    # Feature 4: closest distance from any invader to any food we are defending.
    invader_food_distances = []
    for invader_position in invader_positions.values():
        for food_position in defended_food:
            distance = agent._distances.get_distance(invader_position, food_position)
            if (distance is not None):
                invader_food_distances.append(distance)

    if (len(invader_food_distances) > 0):
        features['invader_distance_to_food'] = min(invader_food_distances)

    return features


def get_defended_food(
        state: pacai.capture.gamestate.GameState,
        agent_index: int,
        ) -> set[pacai.core.board.Position]:
    defended_food = set()
    team_modifier = state._team_modifier(agent_index)

    for food_position in state.board.get_marker_positions(pacai.pacman.board.MARKER_PELLET):
        if (state._team_side(position = food_position) == team_modifier):
            defended_food.add(food_position)

    return defended_food
