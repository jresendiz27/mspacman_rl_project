import keras
from keras.src.optimizers import Adam


def build_dqn(lr, n_actions, input_dims, fc1_dims=64, fc2_dims=128, fc3_dims=256):
    model = keras.Sequential(
        [
            keras.Input(shape=input_dims),
            keras.layers.Conv2D(fc1_dims, strides=4, kernel_size=8, activation="relu"),
            keras.layers.Conv2D(fc2_dims, strides=2, kernel_size=4, activation="relu"),
            keras.layers.Conv2D(fc3_dims, strides=1, kernel_size=3, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(fc3_dims * 2, activation="relu"),
            keras.layers.Dense(n_actions),
        ]
    )
    model.summary()
    model.compile(loss="mse",
                  optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])
    return model
