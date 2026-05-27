import pacai.core.agentinfo
import pacai.util.alias
import pacai.agents.greedy
import typing
import pacai.search.distance
import pacai.core.gamestate
import pacai.core.features
import pacai.core.action
import pacai.core.agent
import pacai.capture.gamestate
import pacai.core.board
import pacai.pacman.board

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    agent1_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.InitialReflexOffensive")
    agent2_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.DefensiveAgent")

    return [agent1_info, agent2_info]



# first start with two basic offense defense greedy based agents using the distance precomputer
class InitialReflexOffensive(pacai.agents.greedy.GreedyFeatureAgent):
    def __init__(self, **kwargs: typing.Any):
        kwargs['feature_extractor_func'] = reflex_offensive_features_extractor
        super().__init__(**kwargs)
        self.precomputer = pacai.search.distance.DistancePreComputer()
        self.weights['score'] = 50.0
        self.weights['closest_food'] = -2.0
        self.weights['closest_enemy'] = -10.0
        self.weights['nearest_capsule'] = -1.0
        self.weights['on_home_side'] = -5.0

    def game_start(self, initial_state: pacai.core.gamestate.GameState):
        self.precomputer.compute(initial_state.board)


def reflex_offensive_features_extractor(
        state: pacai.capture.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **_kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    
    agent = typing.cast(InitialReflexOffensive, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)
    
    distance_cache = agent.precomputer

    features = pacai.core.features.FeatureDict()

    features['score'] = state.get_normalized_score(agent.agent_index)

    # important featuers for offense
    # 1: distance to nearest power pellet WEIGHT +
    # 2: distance to nearest food pellet WEIGHT +
    # 3: number of food pellets in some N x N area around the nearest pellet (could take as proportion of filled area 0..1) and weight
    # 4: score (make sure to take account pos-neg differences based on red or blue side) (may not need as other features may help encode)
    # 5: distance to nearest ghost FLIP WEIGHT ON SCARED (should maybe use logarithmic weight)
    # 6: how many enemy agents are on the same side of the board
    # 7: is currently scared (only matters if we are on the defending side)
    # 8 side of the board

    current_pos = state.get_agent_position(agent.agent_index)
    if current_pos is None:
        return features

    opposing_positions: dict[int, pacai.core.board.Position] = state.get_opponent_positions(agent_index=agent.agent_index)

    # find closest enemy agent, hard cutoff limit at 10 tiles for including the features
    closest_enemy: tuple[int | None, float | None] = (None, None)  # idx, distance
    for idx, pos in opposing_positions.items():
        distance = distance_cache.get_distance(current_pos, pos)
        if distance is None:
            continue
        if closest_enemy[1] is None or distance <= closest_enemy[1]:
            closest_enemy: tuple[int | None, float | None]= (idx, distance)


    # now using the closest enemy, if it is a ghost but not scared, weigh it (-)
    if closest_enemy[0] is not None and closest_enemy[1] is not None:
        if state.is_ghost(closest_enemy[0]) and not state.is_scared(agent_index = closest_enemy[0]):
            if closest_enemy[1] <= 8:
                features['closest_enemy'] = -1 * closest_enemy[1]
        elif state.is_ghost(closest_enemy[0]) and state.is_scared(agent_index = closest_enemy[0]):
            features['closest_enemy'] = closest_enemy[1]


    # next deal with the food based  feature
    food_positions: set[pacai.core.board.Position] = state.get_food(agent_index=agent.agent_index)
    closest_food_distance = get_nearest_distance(food_positions, distance_cache, current_pos)

    if closest_food_distance is not None:
        features['closest_food'] = closest_food_distance

    # make sure that if both enemy agents are on our side of the board, all in on the nearest food
    has_defender = False
    for idx, pos in opposing_positions.items():
        if state.is_ghost(agent_index=idx):
            has_defender = True
            break
    
    if not has_defender and features.get('closest_food', None) is not None:
        features['closest_food'] = features['closest_food'] * 5  # random 5 for now but we will just have to rip it
    
    # check if on home of the board
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    my_mod = state._team_modifier(agent.agent_index)
    power_capsule_positions = []
    for pos in state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE):
        if state._team_side(position=pos) != my_mod:
            power_capsule_positions.append(pos)

    closest_capsule_distance = get_nearest_distance(power_capsule_positions, distance_cache, current_pos)

    if closest_capsule_distance is not None:
        features['nearest_capsule'] = closest_capsule_distance

    return features

def get_nearest_distance(
        position_collection: set[pacai.core.board.Position] | list[pacai.core.board.Position],
        distance_cache: pacai.search.distance.DistancePreComputer,
        compare: pacai.core.board.Position,
        max_distance: float | None = None
    ) -> float | None:
    closest_pos = None
    for pos in position_collection:
        distance = distance_cache.get_distance(compare, pos)
        if distance is None:
            continue
        if max_distance is not None and distance > max_distance:
            continue
        if closest_pos is None or distance < closest_pos:
            closest_pos = distance
    return closest_pos
    
SAFE_INVADER_DISTANCE = 8

class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    def __init__(self, **kwargs: typing.Any):
        # init the feature extractor func via the kwargs using my modified feature extraction agents
        kwargs['feature_extractor_func'] = extract_defensive_features
        super().__init__(**kwargs)
        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        """ Precompute distances. """

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
    # start with all of the invader positions
    invader_positions = state.get_invader_positions(agent_index = agent.agent_index)
    defended_food = get_defended_food(state, agent.agent_index)
    is_scared = state.is_scared(agent_index = agent.agent_index)

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
                # set hard limit on when to be run away from invaders when scared
                features['distance_to_invader'] = SAFE_INVADER_DISTANCE - min(
                        closest_invader,
                        SAFE_INVADER_DISTANCE)
            else:
                features['distance_to_invader'] = closest_invader

    # sum of all the distances from us to the food to defend
    food_distance_sum = 0
    for food_position in defended_food:
        distance = agent._distances.get_distance(current_position, food_position)
        if (distance is not None):
            food_distance_sum += distance

    features['food_distance_sum'] = food_distance_sum

    # closest distance from any invader to any food we are defending
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