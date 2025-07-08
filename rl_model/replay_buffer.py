import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.discrete = discrete

        # ¿ESTE TIPO DE ESTRUCTURA EN LA MEMORIA SEGUIRA SIENDO DE UTILIDAD?
        # En el caso del ambiente Cart Pole, los ambientes son del tipo [3.23, 12.45, 0.13, 24]
        # y debemos almacenar muchas filas de ese tipo. Sin embargo, para los juegos de atari, tenemos imágenes en tres canales
        # cada una de ellas es un estado, por lo tanto, es probable que esta definición de memoria deba ajustarse...

        # **************************************************************************************************************************
        states_shape = (self.mem_size,) + self.input_shape
        self.state_memory = np.zeros(states_shape)
        self.new_state_memory = np.zeros(states_shape)

        # **************************************************************************************************************************

        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    # ¿Valdrá la pena agregar la bander truncated como un nuevo parametro en este método?
    # **************************************************************************************************************************
    def store_transition(self, state, action, reward, state_, done, truncated):
        # **************************************************************************************************************************
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward

        # Aqui podemos hacer los ajustes considerando el uso de la bandera "truncated" de Gymnasium
        # **************************************************************************************************************************
        self.terminal_memory[index] = 1 - int(done or truncated)
        # **************************************************************************************************************************
        actions = None
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = actions
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
