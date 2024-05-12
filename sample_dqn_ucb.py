from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn_ucb
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


print('AGENT DQN UCB')
for bsuite_id in sweep.SWEEP:
     print("\nTESTING BSUITE ID", bsuite_id, type(bsuite_id))
     SAVE_PATH = './logs/dqn_ucb'
     cur_env = bsuite.load_and_record(bsuite_id, save_path=SAVE_PATH, overwrite=True)
     cur_ag = dqn_ucb.default_agent(obs_spec=cur_env.observation_spec(), action_spec=cur_env.action_spec())
     experiment.run(cur_ag, cur_env, num_episodes=cur_env.bsuite_num_episodes)
