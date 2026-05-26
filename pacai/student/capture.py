import pacai.core.agentinfo
import pacai.util.alias
import pacai.agents.greedy
import typing
import pacai.search.distance
import pacai.core.gamestate
import pacai.core.features
import pacai.core.action
import pacai.core.agent

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    agent1_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)
    agent2_info = pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_DUMMY.long)

    return [agent1_info, agent2_info]



# first start with two basic offense defense greedy based agents using the distance precomputer


class InitialReflexOffensive(pacai.agents.greedy.GreedyFeatureAgent):
    def __init__(self, **kwargs: typing.Any):
        # init the feature extractor func via the kwargs using my modified feature extraction agents

        super().__init__(**kwargs)
        self.precomputer = pacai.search.distance.DistancePreComputer()

    def game_start(self, initial_state: pacai.core.gamestate.GameState):
        self.precomputer.compute(initial_state.board)


def ReflexOffensiveExtractor(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **_kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    
    agent = typing.cast(InitialReflexOffensive, agent)
    
    distance_cache = agent.precomputer

    # important featuers for offense 
    # 1: distance to nearest power pellet WEIGHT +
    # 2: distance to nearest food pellet WEIGHT +
    # 3: number of food pellets in some N x N area around the nearest pellet (could take as proportion of filled area 0..1) and weight
    # 4: score (make sure to take account pos-neg differences based on red or blue side)
    # 5: distance to nearest ghost FLIP WEIGHT ON SCARED (should maybe use logarithmic weight)
    # 6: how many enemy agents are on teh same side of the board