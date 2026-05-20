import os

import edq.testing.unittest
import edq.util.dirent

import pacai.capture.bin
import pacai.core.agentinfo
import pacai.util.alias

THIS_PATH: str = os.path.realpath(__file__)

class GameTest(edq.testing.unittest.BaseTest):
    """ Test specifics for capture games. """

    def test_load_randomboard_replay(self):
        """ Test loading a replay that has a random board. """

        temp_dir = edq.util.dirent.get_temp_dir(prefix = 'pacai-test-')
        replay_path = os.path.join(temp_dir, 'test.replay')

        expected_score = -10

        # Run a short capture game and save the replay.
        argv = [
            '--seed', '4',
            '--quiet',
            '--board', 'random-6',
            '--red', 'capture-team-baseline',
            '--blue', 'capture-team-dummy',
            '--ui', 'null',
            '--max-turns', '80',
            '--save-path', replay_path,

        ]
        _, results = pacai.capture.bin.main(argv = argv)

        self.assertEqual(expected_score, results[0].score)

        # Replay the game and get the same result.
        argv = [
            '--quiet',
            '--ui', 'null',
            '--replay-path', replay_path,

        ]
        _, results = pacai.capture.bin.main(argv = argv)

        self.assertEqual(expected_score, results[0].score)

    def test_team_movedelay(self):
        """ Test that a team cannot set their own move delay. """

        # Game should end close to a tie.
        # A team with the set move delay should always win against a normal team.
        expected_score = 40

        argv = [
            '--seed', '4',
            '--quiet',
            '--red', f"{THIS_PATH}:_create_team_movedelay",
            '--blue', 'capture-team-baseline',
            '--ui', 'null',
            '--max-turns', '100',
        ]
        _, results = pacai.capture.bin.main(argv = argv)

        self.assertEqual(expected_score, results[0].score)

def _create_team_movedelay() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Create a team that tries to override its own move delay.
    """

    return [
        pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_OFFENSIVE.long, move_delay = 10),
        pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_DEFENSIVE.long, move_delay = 10),
        pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_OFFENSIVE.long, move_delay = 10),
        pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_DEFENSIVE.long, move_delay = 10),
        pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_OFFENSIVE.long, move_delay = 10),
        pacai.core.agentinfo.AgentInfo(name = pacai.util.alias.AGENT_CAPTURE_DEFENSIVE.long, move_delay = 10),
    ]
