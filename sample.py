from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn_ucb
from bsuite.baselines.tf import boot_dqn
from bsuite.baselines.tf import boot_dqn_squarecb
from bsuite.baselines.tf import boot_dqn_ucb
from bsuite.experiments import summary_analysis
from bsuite.experiments.bandit_noise import analysis as bandit_noise_analysis
from bsuite.logging import csv_load
from bsuite import sweep

import bsuite.logging
import bsuite.experiments
import warnings
import numpy as np
import pandas as pd
import plotnine as gg
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing_x=0.5, panel_spacing_y=0.5)
warnings.filterwarnings('ignore')

envs = [
      "bsuite_id_-_bandit_noise-0.csv",
      "bsuite_id_-_bandit-0.csv",
      "bsuite_id_-_cartpole_noise-11.csv",
      "bsuite_id_-_cartpole-0.csv",
      "bsuite_id_-_deep_sea_stochastic-2.csv",
      "bsuite_id_-_umbrella_length-19.csv",
]

env_cat = [
      'Bandit-Noise',
      'Bandit',
      'Cartpole-Noise',
      'Cartpole',
      'Deep-Sea-Stochastic',
      'Umbrella-Length'
]

agent_paths = ["dqn", "dqn_squarecb", "dqn_ucb"]

for i in range(len(envs)):
      env_name = envs[i]
      print('\nENV_NAME:', env_name)
      plt.figure(figsize=(10, 6))
      for agent_path in agent_paths:
            print('AGENT:', agent_path)
            df = pd.read_csv("./logs" + "/" + agent_path + "/" + env_name)
            plt.plot(df['steps'], df['episode_return'], marker='o', linestyle='-', label=agent_path)
      plt.title(env_cat[i] + ' Steps vs Episode Return')
      plt.xlabel('Steps')
      plt.ylabel('Episode Return')
      plt.legend()
      plt.grid(True)
      plt.show()


# print("STARTING DATA ANALYSIS\n")
# for bsuite_id in sweep.SWEEP:
#       env = bsuite.load_from_id(bsuite_id)
#       print('bsuite_id={}, settings={}, num_episodes={}'
#         .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

# ADD AGENTS NAMES HERE
# agent_names = ['dqn', 'dqn_squarecb', 'dqn_ucb']
# def getEnvAgent(bsuite_id, ag):
#       SAVE_PATH = './logs/' + ag
#       ENV = bsuite.load_and_record(bsuite_id, save_path=SAVE_PATH, overwrite=True)
      
#       #baseline
#       AGENT = boot_dqn.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())

#       if ag == 'boot_dqn':
#             AGENT = boot_dqn.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())
#       elif ag == "boot_dqn_squarecb":
#             AGENT = boot_dqn_squarecb.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())
#       elif ag == "boot_dqn_ucb":
#             AGENT = boot_dqn_ucb.default_agent(obs_spec=ENV.observation_spec(), action_spec=ENV.action_spec())
      
#       # ADD AGENT INFORMATION HERE
#       return ENV, AGENT

# for ag_name in agent_names:
#       print("\n-------------------------------------")
#       print('AGENT', ag_name)
#       # for bsuite_id in sweep.SWEEP:
#       for bsuite_id in sweep.SWEEP:
#             print("\nTESTING BSUITE ID", bsuite_id, type(bsuite_id))
#             cur_env, cur_ag = getEnvAgent(bsuite_id, ag_name)
#             experiment.run(cur_ag, cur_env, num_episodes=cur_env.bsuite_num_episodes)

# experiments = {}
# for ag_name in agent_names:
#       ag_path = './logs/' + ag_name
#       experiments[ag_name] = ag_path

# DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
# BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)
# BSUITE_SUMMARY = summary_analysis.ave_score_by_tag(BSUITE_SCORE, SWEEP_VARS)
# print("BSUITE_SCORE:", BSUITE_SCORE)
# print("BSUITE_SUMMARY:", BSUITE_SUMMARY)
# bandit_noise_df = DF[DF.bsuite_env == 'bandit_noise'].copy()
# summary_analysis.plot_single_experiment(BSUITE_SCORE, 'bandit_noise', SWEEP_VARS)
# bandit_noise_analysis.plot_average(bandit_noise_df, SWEEP_VARS) 
# __radar_fig__ = summary_analysis.bsuite_radar_plot(BSUITE_SUMMARY, SWEEP_VARS)