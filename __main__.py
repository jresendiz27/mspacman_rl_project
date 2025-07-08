from rl_model.agent import Agent


def run_agent():
    pacman_agent = Agent(
        alpha=0,
        gamma=0.9,
        epsilon=0.1,
        n_actions=10,
        input_dims=(80,80,3),
        mem_size=2000,
        epsilon_end=0.001,
        batch_size=100,
    )



if __name__ == '__main__':
    print("Main")