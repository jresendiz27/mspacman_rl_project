import logging

import tensorflow as tf

from rl_model.dqn_network import build_dqn
from rl_model.replay_buffer import ReplayBuffer
import numpy as np

import keras.models as keras_models

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000, fname='dqn_model.keras', model_verbose=False):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        #Voy a agregar un atributo path, para indicar la carpeta donde quiero guardar mi modelo entrenado
        self.path = "./datasets/pacman_model/"

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 128, 64)
        self.model_verbose = model_verbose

    def adjust_state(self, state):
        adjusted_state = tf.convert_to_tensor(state, dtype=tf.float32)
        # adjusted_state = tf.expand_dims(adjusted_state, 0)
        return adjusted_state

    def remember(self, state, action, reward, new_state, done, truncated):
        self.memory.store_transition(state, action, reward, new_state, done, truncated)

    def choose_action(self, state):
        #**************************************************************************************************************************
        #Al hacer el cambio en el ambiente y usar datos con dimensiones distintas, es necesario ajustar tanto las consultas a la memoria
        #como los llamados a la red neuronal

        state = state[np.newaxis, :]  #Puede que esta línea les marque un error
        logging.info(f"Chosen state! state: {state}")
        #**************************************************************************************************************************
        rand = np.random.random()
        if rand < self.epsilon:
            logging.info("Taking random action!")
            action = np.random.choice(self.action_space)
        else:
            #**************************************************************************************************************************
            #Es dentro del else donde se usa la variable $state$, por lo tanto, conviene hacer aqui los ajustes al dato que le enviamos
            #al modelo $q_eval$.
            #Revisen la documentación de numpy, en: https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
            #Para tener una mejor idea de qué cambios hacerle a esta variable
            logging.info("Checking the memories!")
            logging.info(f"State shape: {state.shape}")
            actions = self.q_eval.predict(self.adjust_state(state), verbose=self.model_verbose)
            action = np.argmax(actions)
        #**************************************************************************************************************************
        logging.info(f"Predicted value: {action}")
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        actions_indices = np.dot(action, action_values)

        #**************************************************************************************************************************
        #En estos llamados al modelo, también es necesario ajustar el dato para propagarlo por la red. De hecho, el cambio podría ser
        #el mismo que en el método choose_action
        #Revisen la documentación de numpy, en: https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        #Para tener una mejor idea de qué cambios hacerle a esta variable

        q_eval = self.q_eval.predict(self.adjust_state(state), verbose=self.model_verbose)
        q_next = self.q_eval.predict(self.adjust_state(new_state), verbose=self.model_verbose)

        #**************************************************************************************************************************
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions_indices] = reward + \
                                                 self.gamma * np.max(q_next, axis=1) * done

        _ = self.q_eval.fit(self.adjust_state(state), q_target, verbose=self.model_verbose)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
                                                          self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.path + self.model_file)

    def load_model(self):
        self.q_eval = keras_models.load_model(self.path + self.model_file)
        print("Model loaded")

