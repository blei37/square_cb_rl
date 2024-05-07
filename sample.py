import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.baselines.tf import boot_dqn
from bsuite import sweep

print("STARTING TEST SCRIPT\n")
# for bsuite_id in sweep.SWEEP:
#       env = bsuite.load_from_id(bsuite_id)
#       print('bsuite_id={}, settings={}, num_episodes={}'
#         .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

#ADD AGENTS NAMES HERE
agents = ['dqn']
def getEnvAgent(bsuite_id, agent):
      SAVE_PATH_BASE = './logs/tests/'

      #baseline
      SAVE_PATH = SAVE_PATH_BASE + str(agent)
      ENV = bsuite.load_and_record(bsuite_id, save_path=SAVE_PATH, overwrite=True)
      AGENT = boot_dqn.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())
      if agent == 'dqn':
            SAVE_PATH = SAVE_PATH_BASE + agent
            ENV = bsuite.load_and_record(bsuite_id, save_path=SAVE_PATH, overwrite=True)
            AGENT = boot_dqn.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())
      
      # ADD AGENT INFORMATION HERE
      return ENV, AGENT



for agent in agents:
      print("\n-------------------------------------")
      print('AGENT ', agent)
      for bsuite_id in sweep.SWEEP:
            print("\nTESTING BSUITE ID", bsuite_id)
            env, agent = getEnvAgent(bsuite_id, agent)
            experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)

# SAVE_PATH_DQN = './logs/test_boot'
# env = bsuite.load_and_record("deep_sea/10", save_path=SAVE_PATH_DQN, overwrite=True)
# agent = boot_dqn.default_agent(
#       obs_spec=env.observation_spec(),
#       action_spec=env.action_spec()
# )
# experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)