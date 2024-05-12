import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn_ucb, dqn_squarecb, dqn
from bsuite.baselines.tf import boot_dqn
from bsuite.baselines.tf import boot_dqn_squarecb
from bsuite.baselines.tf import boot_dqn_ucb
from bsuite import sweep
import bsuite.logging
import bsuite.experiments
from bsuite.logging import csv_load

episodes = 2
def test_run(save_path, bsuite_id, agent):
    env = bsuite.load_and_record(bsuite_id, save_path=save_path, overwrite=True)
    agent = agent.default_agent(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec()
    )
    experiment.run(agent, env, num_episodes=episodes)

exps=[]
for name in sweep.SWEEP:
    n = name.split('/')[0]
    if n not in exps:
        exps.append(n)
print('total envs:', len(exps))

for i,n in enumerate(exps):
    if 'mnist' in n:
        continue
    print('testing env:',i, n)
    SAVE_PATH = './logs/envtest'
    test_run(SAVE_PATH, n+"/0", dqn)


# # normal dqn
# SAVE_PATH = './logs/test_dqn_cartpole'
# test_run(SAVE_PATH, "cartpole/0", dqn)
# SAVE_PATH = './logs/test_ucb_cartpole'
# test_run(SAVE_PATH, "cartpole/0", dqn_ucb)
# SAVE_PATH = './logs/test_squareucb_cartpole'
# test_run(SAVE_PATH, "cartpole/0", dqn_squarecb)

# # bootdqn
# SAVE_PATH = './logs/test_boot_dqn_cartpole'
# test_run(SAVE_PATH, "cartpole/0", boot_dqn)
# SAVE_PATH = './logs/test_boot_ucb_cartpole'
# test_run(SAVE_PATH, "cartpole/0", boot_dqn_ucb)
# SAVE_PATH = './logs/test_boot_squareucb_cartpole'
# test_run(SAVE_PATH, "cartpole/0", boot_dqn_squarecb)