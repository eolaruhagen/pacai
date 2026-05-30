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

    agent1_info = pacai.core.agentinfo.AgentInfo(
            name = f"{__name__}.UnifiedAgent",
            my_mode_var = 'agent_1_last_turn_mode',
            teammate_mode_var = 'agent_2_last_turn_mode')
    agent2_info = pacai.core.agentinfo.AgentInfo(
            name = f"{__name__}.UnifiedAgent",
            my_mode_var = 'agent_2_last_turn_mode',
            teammate_mode_var = 'agent_1_last_turn_mode')

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
        self.weights['off_on_home_side'] = -40.0
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

    agent = typing.cast(UnifiedAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    distance_cache = agent._distances

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

    features['off_on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))
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
        self.weights['def_on_home_side'] = 1300.0
        self.weights['stopped'] = -95.0

    def game_start(self, initial_state: pacai.core.gamestate.GameState):
        self._distances.compute(initial_state.board)
        
def extract_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **_kwargs: typing.Any) -> pacai.core.features.FeatureDict:

    agent = typing.cast(UnifiedAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if (current_position is None):
        # We are dead and waiting to respawn.
        return features

    # Note the side of the board we are on.
    features['def_on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

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

    enemies = list(state.get_opponent_positions(agent_index = agent.agent_index).values())


    # - `distance_to_threatened_capsule`
    #   - A capsule is threatened if an invader is within `CAPSULE_GUARD_RANGE = 6`.
    #   - If the defender is not scared, prioritize moving toward the threatened capsule instead of directly chasing the invader.
    #   - If the defender is scared, scared-distance behavior takes precedence.
    # - `distance_from_attacker_to_powerpellet`
    #   - only weigh this feature heavely if the offensive agent is close to the power pellet.
    #   - if the offensive agent is close to the power pellet, and the defensive agent is close to the power pellet, then the defensive agent should move toward the power pellet and guard it if and only if the defender is close to the power pellet.
    #   - we get the distance of the attacker to its nearist power pellet, and the distance of the defender to that same power pellet.
    #   - first, if the attacker is not even near its nearist power pellet, we can just not have this feature activate, and we only want to activate this feature if the attacker is within a certain range of its nearest power pellet. If the attacker is close, then we determine the differnence in distnaces between the defender and the attacker relative to the power pellet. if we are close, the reward is positive, and if the attacker is close and we are far, the reward is negative.
    # protect a power pellet if they are close to it
    pellet_range = 6
    power_pellets = []
    my_side = state._team_modifier(agent.agent_index)
    for pellet in state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE):
        if (state._team_side(position = pellet) == my_side):
            power_pellets.append(pellet)

    closest_pellet = None
    attacker_dist = None
    for pellet in power_pellets:
        dist = get_nearest_distance(enemies, agent._distances, pellet)
        if (dist is not None and dist <= pellet_range
                and (attacker_dist is None or dist < attacker_dist)):
            closest_pellet = pellet
            attacker_dist = dist

    if (closest_pellet is not None and attacker_dist is not None and not is_scared):
        my_dist = agent._distances.get_distance(current_position, closest_pellet)
        if (my_dist is not None):
            features['distance_from_attacker_to_powerpellet'] = (attacker_dist - my_dist)

            invader_dist = get_nearest_distance(invader_pos_list, agent._distances, closest_pellet)
            if (invader_dist is not None and invader_dist <= pellet_range):
                features['distance_to_threatened_capsule'] = my_dist

    # - `oscilating_peanlity`
    #   - This feature's goal is to give a higher and higher penalty as the agent oscillates back and forth.
    #   - Have a momentum variable that stores the penalty of the current oscillation of the agent
    #   - We analyze the previous three positions of the agent. If the first position and the last position are the same, that means we just made an oscillation, which would trigger the momentum variable to be multiplied by 1.2. Otherwise if it's not an oscillation, which would be the scenario if position 1 and position 3 are not the same, then we multiply the momentum variable by 0.9. We don't make the momentum variable go below 0.5
    # give more of a penalty if it keeps going back and forth
    loop_penalty = agent.extra_storage.get('loop_penalty', 0.5)
    num_positions = len(agent.last_positions)
    if (agent.extra_storage.get('last_loop_check') != num_positions):
        if (num_positions >= 3 and agent.last_positions[-3] == agent.last_positions[-1]):
            loop_penalty *= 1.2
        else:
            loop_penalty = max(0.5, loop_penalty * 0.9)

        agent.extra_storage['loop_penalty'] = loop_penalty
        agent.extra_storage['last_loop_check'] = num_positions

    if (num_positions >= 2 and current_position == agent.last_positions[-2]):
        features['oscilating_peanlity'] = loop_penalty

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

    agent = typing.cast(UnifiedAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    current_mode = globals()[agent.my_mode]
    teammate_mode = agent.read_teamate_mode()

    invaders = state.get_invader_positions(agent_index=agent.agent_index)
    num_invaders = len(invaders)

    # check if we want to prefer attacking by being in the offense window
    in_post_kill_window = False
    if agent.turns_since_kill is not None:
        if agent.turns_since_kill < POST_KILL_OFFENSIVE_WINDOW:
            in_post_kill_window = True

    if current_mode is not None and agent.turns_in_mode < MIN_TURNS_PER_MODE:
        mode = current_mode

    # defend when we have invaders and no existing defender
    elif num_invaders > 0 and teammate_mode != 'defensive':
        mode = 'defensive'

    elif in_post_kill_window:
        mode = 'offensive'
    elif num_invaders == 0 and teammate_mode != 'offensive':
        mode = 'offensive'
    else:
        mode = 'defensive'  # default we could tune, but push towards offensive

    agent.write_used_mode(mode)

    if mode == 'offensive':
        return reflex_offensive_features_extractor(state, action, agent=agent)
    return extract_defensive_features(state, action, agent=agent)


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

        self.border_crossings: set[pacai.core.board.Position] = set()
        self.team_side = typing.Literal['left', 'right']

        # defensive mode features
        self.weights['distance_to_invader'] = -45.0
        self.weights['distance_to_invader_when_scared'] = -75.0
        self.weights['food_distance_sum'] = -2.0
        self.weights['num_invaders_on_same_side'] = -6500.0
        self.weights['invader_distance_to_food'] = 30.0
        self.weights['def_on_home_side'] = 1300.0
        self.weights['stopped'] = -95.0
        self.weights['distance_to_threatened_capsule'] = -45.0
        self.weights['distance_from_attacker_to_powerpellet'] = 45.0
        self.weights['oscilating_peanlity'] = -15.0 

        # offensive mode features go here
        self.precomputer = pacai.search.distance.DistancePreComputer()
        self.threat_detection_range = 10
        self.safe_food_distance = 5
        self.weights['score'] = 100.0
        self.weights['distance_to_food'] = -22.0
        self.weights['distance_to_non_defended_food'] = -3.5
        self.weights['distance_to_threat'] = 10
        self.weights['distance_to_prey'] = -40.0
        self.weights['nearest_capsule'] = -6.0
        self.weights['off_on_home_side'] = -40.0
        self.weights['on_home_side_unsafe'] = -50.0
        self.weights['have_scared_enemy'] = 1100
        self.weights['oscilating_action'] = -3.0
        self.weights['food_eaten_by_action'] = 1200.0
        self.weights['food_cluster_nearby'] = 5.0


        # shared features
        self.weights['distance_to_teamate'] = 10
        

        # stateful vars for conditional logic on feature extraction
        self.turns_in_mode: int = 0
        self.num_kills: int = 0
        self.turns_since_kill: int | None = None  # init to none when no kills yet
        self.prev_enemy_positions: dict[int, pacai.core.board.Position] = dict()
        # represent the last seen mode for current and team agent (from the last turn)
        self.my_mode = my_mode_var
        self.teammate_mode = teammate_mode_var


    def game_start(self, initial_state: pacai.core.gamestate.GameState):
        self._init_border_crossing_positions(initial_state)
        self._distances.compute(initial_state.board)
        state = typing.cast(pacai.capture.gamestate.GameState, initial_state)
        self.prev_enemy_positions = state.get_opponent_positions()
        super().game_start(initial_state)
    

    # helper methods for reading module level shared state variables
    def read_teamate_mode(self) -> typing.Literal['offensive' , 'defensive'] | None:
        return globals()[self.teammate_mode]
    
    def write_used_mode(self, mode: typing.Literal['offensive' , 'defensive'] | None):
        globals()[self.my_mode] = mode

    def _init_border_crossing_positions(self, initial_state: pacai.core.gamestate.GameState):
        team_side = initial_state._team_modifier(agent_index=self.agent_index)

        width = initial_state.board.width
        height_bound = initial_state.board.height
        center_column = None

        if team_side == -1:
            self.team_side = 'left'
            center_column = (width // 2) - 1
        elif team_side == 1:
            self.team_side = 'right'
            center_column = (width // 2)

        if center_column is None:
            return
        # now initialize border crossings iterating through non-wall positions on boundary_column
        for pos_y in range(height_bound):
            pos = pacai.core.board.Position(pos_y, center_column)
            if not initial_state.board.is_wall(pos):
                self.border_crossings.add(pos)
        return

    def _find_kill(self, state: pacai.core.gamestate.GameState) -> bool:
        state = typing.cast(pacai.capture.gamestate.GameState, state)
        # skip if dead, waste of time.
        current_pos = state.get_agent_position(self.agent_index)
        if current_pos is None:
            return False
        # otherwise, if my current position is the prev pos of enenmy, we found a kill.
        # more complex logic (enemy moves into us) to be handled later
        # current_opponent_positions: dict[int, pacai.core.board.Position] = state.get_opponent_positions()
        for pos in self.prev_enemy_positions.values():
            if current_pos == pos:
                return True
        return False

    # this one is simply for updating teh stateful vars on the num kills and turns since kill
    def update_stateful_vars(self, state: pacai.core.gamestate.GameState):
        state = typing.cast(pacai.capture.gamestate.GameState, state)
        if self._find_kill(state):
            self.num_kills += 1
            self.turns_since_kill = 0
        elif self.turns_since_kill is not None:
            # bumpt the turns since kill
            self.turns_since_kill += 1

    def distance_to_teamate(self, state: pacai.core.gamestate.GameState) -> float | None:
        capture_state = typing.cast(pacai.capture.gamestate.GameState, state)
        teamate_pos = list(capture_state.get_ally_positions().values())[0]
        current_pos = state.get_agent_position(agent_index=self.agent_index)
        if teamate_pos is None:
            return None
        return self._distances.get_distance(current_pos, teamate_pos)



    def get_action(self, state: pacai.core.gamestate.GameState) -> pacai.core.action.Action:
        self.update_stateful_vars(state)
        last_mode = globals()[self.my_mode]
        action = super().get_action(state)
        # flow in get_action can change self.my_mode use this for internal turns in mode
        if last_mode == globals()[self.my_mode]:
            self.turns_in_mode += 1
        else:
            self.turns_in_mode = 0

        self.prev_enemy_positions = state.get_opponent_positions()
        return action
    


        



    

        
    
        

    
