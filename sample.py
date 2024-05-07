import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.baselines.tf import boot_dqn
from bsuite import sweep
from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis

print("STARTING TEST SCRIPT\n")
# for bsuite_id in sweep.SWEEP:
#       env = bsuite.load_from_id(bsuite_id)
#       print('bsuite_id={}, settings={}, num_episodes={}'
#         .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

# ADD AGENTS NAMES HERE
agent_names = ['dqn']
def getEnvAgent(bsuite_id, ag):
      SAVE_PATH = './logs/' + ag

      #baseline
      ENV = bsuite.load_and_record(bsuite_id, save_path=SAVE_PATH, overwrite=True)
      AGENT = boot_dqn.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())

      if ag == 'dqn':
            ENV = bsuite.load_and_record(bsuite_id, save_path=SAVE_PATH, overwrite=True)
            AGENT = boot_dqn.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())
      
      # ADD AGENT INFORMATION HERE
      return ENV, AGENT

for ag_name in agent_names:
      print("\n-------------------------------------")
      print('AGENT ', ag_name)
      for bsuite_id in sweep.SWEEP:
            print("\nTESTING BSUITE ID", bsuite_id)
            cur_env, cur_ag = getEnvAgent(bsuite_id, ag_name)
            experiment.run(cur_ag, cur_env, num_episodes=10)

experiments = {}
for ag_name in agent_names:
      ag_path = './logs/' + ag_name
      experiments[ag_name] = ag_path
DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
print(BSUITE_SCORE)

# SAVE_PATH_DQN = './logs/test_boot'
# env = bsuite.load_and_record("deep_sea/9", save_path=SAVE_PATH_DQN, overwrite=True)
# agent = boot_dqn.default_agent(
#       obs_spec=env.observation_spec(),
#       action_spec=env.action_spec()
# )
# experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)

# SAVE_PATH_DQN = './logs/test_boot1'
# for bsuite_id in sweep.SWEEP:
#   env = bsuite.load_and_record(bsuite_id, save_path=SAVE_PATH_DQN, overwrite=True)
#   agent = dqn.default_agent(
#       obs_spec=env.observation_spec(),
#       action_spec=env.action_spec()
#   )
#   experiment.run(agent, env, num_episodes=10)