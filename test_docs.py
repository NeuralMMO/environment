from pdb import set_trace as T
#Core API
from forge.blade.core import Realm
from forge.blade import IO

#Demo baselines
from experiments import Experiment, Config
from forge.ethyr.torch.policy import baseline

#Define an experiment configuration
config = Experiment('demo', Config).init(TEST=True)

#Initialize the environment and policy
env                        = Realm(config)
obs, rewards, dones, infos = env.reset()
policy                     = baseline.IO(config)

for i in range(5):
   obs, rewards, dones, infos = env.step({})

#Process observations
packet, deserialize, _    = IO.inputs(obs, rewards, dones, config)
flat, lookup = policy.input(packet)

#Select actions
policy.output(packet, flat, lookup)
actions      = IO.outputs(packet, deserialize)

#Submit actions
nxtObs, rewards, dones, info = env.step(actions)

#(s, a, r) tuple + rollout boundaries
print(obs, actions, rewards, dones)
