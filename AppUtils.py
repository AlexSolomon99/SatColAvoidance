

def play_game_manually(game_env):

    # resetting the game and getting the firs obs
    obs, _ = game_env.reset()

    # condition for the game to be over
    done = False

    # set list of rewards and observations
    rewards = []
    observations = []
    actions = []

    while not done:
        print(f"Step idx: {game_env.unwrapped.time_step_idx}")
        action = list(map(int, input("\nCurrent Action : ").strip().split()))[:game_env.action_space.shape[0]]
        print("Action received from user: ", action)

        obs, reward, terminated, truncated, info = game_env.step(action)

        rewards.append(reward)
        actions.append(action)
        observations.append(obs)

    return rewards, observations, actions


def play_constant_game_manually(game_env, constant_action):

    # resetting the game and getting the firs obs
    obs, _ = game_env.reset()

    # condition for the game to be over
    done = False

    # set list of rewards and observations
    rewards = []
    observations = []
    actions = []

    while not done:

        obs, reward, done, truncated, info = game_env.step(constant_action)

        rewards.append(reward)
        actions.append(constant_action)
        observations.append(obs)

    return rewards, observations, actions
