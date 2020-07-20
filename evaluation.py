# Metrics and evaluation
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    tensor_avg_return = total_return / num_episodes
    num_avg_return = tensor_avg_return.numpy()[0]
    return num_avg_return


def win_rate(environment, policy, num_episodes=10):
    wins = 0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        if episode_return >= 1:
            wins += 1
    return (wins / num_episodes) * 100
