import nmmo
from nmmo.core.config import Medium


def create_config(base, *systems):
    systems   = (base, *systems)
    name      = '_'.join(cls.__name__ for cls in systems)
    conf      = type(name, systems, {})()

    conf.TERRAIN_TRAIN_MAPS = 1
    conf.TERRAIN_EVAL_MAPS  = 1
    conf.IMMORTAL = True
    conf.RENDER   = True

    return conf

def benchmark_config(base, nent, *systems):
    conf = create_config(base, *systems)
    conf.PLAYER_N = nent
    env = nmmo.Env(conf)
    env.reset()

    env.render()
    while True:
        env.step(actions={})
        env.render()

benchmark_config(Medium, 100)

