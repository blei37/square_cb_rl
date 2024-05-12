import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.baselines.tf import boot_dqn
from bsuite.baselines.tf import boot_dqn_squarecb
from bsuite.baselines.tf import boot_dqn_ucb
from bsuite import sweep
import bsuite.logging
import bsuite.experiments
from bsuite.logging import csv_load

def test_run(save_path, bsuite_id, agent):
    env = bsuite.load_and_record(bsuite_id, save_path=save_path, overwrite=True)
    agent = agent.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)

SAVE_PATH = './logs/test_boot_ucb_cartpole'
test_run(SAVE_PATH, "cartpole/0", boot_dqn_ucb)