from gym.envs.registration import register

register(
    id='BSSEnv-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(1, 21))
    }
)

register(
    id='BSSEnvTest-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-data-art',
        'scenarios': list(range(21, 61))
    }
)

register(
    id='BSSEnv-v1',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-art',
        'scenarios': list(range(0, 30))
    }
)

register(
    id='BSSEnvTest-v1',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-art',
        'scenarios': list(range(30, 100))
    }
)

register(
    id='BSSEnv-v2',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-OD-art',
        'scenarios': list(range(0, 30))
    }
)

register(
    id='BSSEnvTest-v2',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'data_set_name': 'actual-poisson-OD-art',
        'scenarios': list(range(30, 100))
    }
)
