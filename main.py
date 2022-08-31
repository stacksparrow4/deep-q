import gym
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent


if __name__ == '__main__':
    n_games = 500

    env = gym.make('CartPole-v1')

    n_inputs = env.reset().shape
    assert len(n_inputs) == 1
    n_inputs = n_inputs[0]
    n_actions = env.action_space.n

    agent = Agent(n_inputs, n_actions, [64, 32])

    scores = []
    losses = []

    for i in range(n_games):
        done = False
        score = 0
        loss = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, _ = env.step(action)
            score += reward
            agent.remember(state, new_state, action, reward, done)
            state = new_state

            loss = agent.train_step()

        losses.append(loss)
        scores.append(score)

        avg_score = np.mean(scores[-5:])
        print("episode: ", i, "score %.2f" %
              score, "average score %.2f" % avg_score, "epsilon %.2f" % agent.eps, "loss %.2f" % loss)

        if avg_score > 190:
            break

    xs = [i+1 for i in range(len(scores))]
    plt.plot(xs, scores, label="scores")
    plt.plot(xs, losses, label="losses")

    plt.legend()
    plt.show()
