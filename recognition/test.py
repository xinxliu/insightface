import sys
import os
sys.path.append(os.path.join('recognition', 'symbol'))
sys.path.append(os.path.join('recognition'))

import fmobilenetV3

from config import config, default, generate_config

network = 'mv3'
dataset = 'emore'
loss = 'arcface'

generate_config(network, dataset, loss)

fc1 = eval(config.net_name).get_symbol()