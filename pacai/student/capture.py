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
import typing

from pacai.util.reflection import Reference


agent_1_last_turn_mode: typing.Literal['offensive' , 'defensive'] | None = None
agent_2_last_turn_mode: typing.Literal['offensive' , 'defensive'] | None = None
team_kills = 0

MIN_TURNS_PER_MODE = 10

# number of turns where attacking is safe after a kill
# this logic may not be immediately invoked, depends on how close agent is
# to a mid-board opening
POST_KILL_OFFENSIVE_WINDOW = 12


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
        self.threat_detection_range = 10
        self.safe_food_distance = 5
        self.weights['score'] = 100.0
        self.weights['distance_to_food'] = -22.0
        self.weights['distance_to_non_defended_food'] = -3.5
        self.weights['distance_to_threat'] = 10
        self.weights['distance_to_prey'] = -40.0
        self.weights['nearest_capsule'] = -6.0
        self.weights['on_home_side'] = -40.0
        self.weights['on_home_side_unsafe'] = -50.0
        self.weights['have_scared_enemy'] = 1100
        self.weights['oscilating_action'] = -3.0
        self.weights['food_eaten_by_action'] = 1200.0
        self.weights['food_cluster_nearby'] = 5.0

    def game_start(self, initial_state: pacai.core.gamestate.GameState):
        self.precomputer.compute(initial_state.board)
        width = initial_state.board.width
        self.threat_detection_range = max(8, width // 3)
        self.safe_food_distance = max(4, width // 5)

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

    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))
    features['on_home_side_unsafe'] = int(state.is_scared(agent_index = agent.agent_index))

    threat_positions = []
    prey_positions = []
    have_scared = False
    opposing_positions: dict[int, pacai.core.board.Position] = state.get_opponent_positions(
            agent_index=agent.agent_index)

    # first start with getting the opposing positinos, but seperate by ghost scared vs not scared for 2  new features
    # this will allow what i had earlier to be seperated into different features with configuable weights
    for idx, pos in opposing_positions.items():
        if state.is_scared(agent_index=idx):
            features['have_scared_enemy'] = 1
            have_scared = True
            prey_positions.append(pos)
        elif state.is_ghost(agent_index=idx):
            threat_positions.append(pos)

    distance_to_threat = get_nearest_distance(
            threat_positions,
            distance_cache,
            current_pos,
            max_distance=agent.threat_detection_range)
    if distance_to_threat is not None:
        features['distance_to_threat'] = distance_to_threat

    distance_to_prey = get_nearest_distance(prey_positions, distance_cache, current_pos)
    if distance_to_prey is not None:
        features['distance_to_prey'] = distance_to_prey

    # next deal with the food based  feature
    food_positions: set[pacai.core.board.Position] = state.get_food(agent_index=agent.agent_index)
    distance_to_food = get_nearest_distance(food_positions, distance_cache, current_pos)
    if distance_to_food is not None:
        features['distance_to_food'] = distance_to_food

    # make sure that if both enemy agents are far enough from the food pellet, its a higher weighted feature (preferably)
    non_defended_food = []
    for food_pos in food_positions:
        nearest_threat_to_food = get_nearest_distance(threat_positions, distance_cache, food_pos)
        if nearest_threat_to_food is None or nearest_threat_to_food > agent.safe_food_distance:
            non_defended_food.append(food_pos)

    distance_to_non_defended_food = get_nearest_distance(
            non_defended_food,
            distance_cache,
            current_pos)
    if distance_to_non_defended_food is not None:
        features['distance_to_non_defended_food'] = distance_to_non_defended_food

    my_mod = state._team_modifier(agent.agent_index)
    power_capsule_positions = []
    for pos in state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE):
        if state._team_side(position=pos) != my_mod:
            power_capsule_positions.append(pos)

    closest_capsule_distance = get_nearest_distance(
            power_capsule_positions,
            distance_cache,
            current_pos)

    if closest_capsule_distance is not None and not have_scared:
        features['nearest_capsule'] = closest_capsule_distance

    # oscilating actions become serious issue on the border btwn the two sides, also helps with finding alternate attack routes
    past_actions = state.get_agent_actions(agent.agent_index)
    if len(past_actions) >= 2:
        if action == state.get_reverse_action(past_actions[-2]):
            features['oscilating_action'] = 1

    # Gaurav changes: reward eating now and moving near groups of food.
    if action != pacai.core.action.STOP and state.is_pacman(agent_index = agent.agent_index):
        if current_pos not in food_positions:
            features['food_eaten_by_action'] = 1

    for food_pos in food_positions:
        distance = distance_cache.get_distance(current_pos, food_pos)
        if distance is not None and distance <= 5:
            features['food_cluster_nearby'] = features.get('food_cluster_nearby', 0) + 1

    return features

def get_nearest_distance(
        position_collection: set[pacai.core.board.Position] | list[pacai.core.board.Position],
        distance_cache: pacai.search.distance.DistancePreComputer,
        compare: pacai.core.board.Position,
        max_distance: float | None = None,
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
        self._distances: pacai.search.distance.DistancePreComputer = (
                pacai.search.distance.DistancePreComputer())
        """ Precompute distances. """

        self.weights['distance_to_invader'] = -45.0
        self.weights['distance_to_invader_when_scared'] = -75.0
        self.weights['food_distance_sum'] = -2.0
        self.weights['num_invaders_on_same_side'] = -6500.0
        self.weights['invader_distance_to_food'] = 30.0
        self.weights['on_home_side'] = 1300.0
        self.weights['stopped'] = -95.0

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

    # changing code for invader positions to use the nearest distance helper
    invader_pos_list = []
    for invader_position in invader_positions.values():
        invader_pos_list.append(invader_position)

    closest_invader = get_nearest_distance(invader_pos_list, agent._distances, current_position)

    if closest_invader is not None:
        if is_scared and closest_invader < SAFE_INVADER_DISTANCE:
            features['distance_to_invader_when_scared'] = closest_invader
        else:
            # still prefer to overwrite this with invader w/ greatest density of food nearby
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


def conditional_feature_extractor(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **_kwargs: typing.Any
) -> pacai.core.features.FeatureDict:
    
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    return {}


class UnifiedAgent(pacai.agents.greedy.GreedyFeatureAgent):
    def __init__(self, 
                 # these two var reads can be configured at setup of
                 my_mode_var: str = 'agent_1_last_turn_mode',
                 teammate_mode_var: str = 'agent_2_last_run_mode',
                **kwargs: typing.Any
                ) -> None:
        kwargs['feature_extractor_func'] = conditional_feature_extractor
        super().__init__(**kwargs)
        self._distances: pacai.search.distance.DistancePreComputer = (
                pacai.search.distance.DistancePreComputer())
        """ Precompute distances. """

        self.my_mode = my_mode_var
        self.teammate_mode = teammate_mode_var

        self.border_crossings: set[pacai.core.board.Position] = set()
        self.team_side = typing.Literal['left', 'right']

        # defensive mode features
        self.weights['distance_to_invader'] = -45.0
        self.weights['distance_to_invader_when_scared'] = -75.0
        self.weights['food_distance_sum'] = -2.0
        self.weights['num_invaders_on_same_side'] = -6500.0
        self.weights['invader_distance_to_food'] = 30.0
        self.weights['on_home_side'] = 1300.0
        self.weights['stopped'] = -95.0

        # offensive mode features go here

    def game_start(self, initial_state: pacai.core.gamestate.GameState):
        self._distances.compute(initial_state.board)
    
    # helper methods for reading module level shared state variables
    def read_teamate_mode(self) -> typing.Literal['offensive' , 'defensive'] | None:
        return globals()[self.teammate_mode]
    
    def write_used_mode(self, mode: typing.Literal['offensive' , 'defensive'] | None):
        globals()[self.my_mode] = mode

    def _init_border_crossing_positions(self, initial_state: pacai.core.gamestate.GameState):




    

