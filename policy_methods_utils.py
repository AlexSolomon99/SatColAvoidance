import torch


def play_game_once(game_env, policy: torch.nn.Module, train: bool, optimizer):
    # setting the policy to the training mode
    if train:
        policy.train()
    else:
        policy.eval()

    # resetting the game and getting the firs obs
    obs, _ = game_env.reset()

    # condition for the game to be over
    done = False

    # set list of rewards
    rewards = []
    log_probs = []

    while not done:
        action_parameters = policy(torch.Tensor(obs))
        action_mean = action_parameters[:3]
        action_std = action_parameters[3:]

        action_normal_distribution = torch.distributions.Normal(action_mean, action_std)
        action = action_normal_distribution.sample()
        log_prob = action_normal_distribution.log_prob(action)

        obs, reward, done, truncated, info = game_env.step(action.item())

        rewards.append(reward)
        log_probs.append(log_prob.reshape(1))

    # print(f"LOF PROBABILITIES: {log_probabilities}")
    log_probabilities = torch.cat(log_probs)

    # get the list of scores for each action
    scores_per_action = NU_NeuralUtils.NeuralNetUtils().compute_discount_rate_reward(list_of_rewards=rewards)

    if train:
        loss = NU_NeuralUtils.NeuralNetUtils().update_policy(scores_per_action, log_probabilities, optimizer)
    else:
        loss = 0

    return loss, len(rewards), scores_per_action
