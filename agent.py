import tensorflow as tf
from keras.api._v2 import keras

from replay import ReplayWrapper

LEARNING_RATE = 0.0005
GAMMA = 0.95

BATCH_SIZE = 512

EPS_MIN = 0.01
EPS_DEC = 5e-4
TARGET_UPDATE_INTERVAL = 100


@tf.function
def choose_action_tf(model: keras.Model, eps: tf.Tensor, state: tf.Tensor, num_actions: int) -> tf.Tensor:
    rng = tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32)

    if rng < eps:
        return tf.random.uniform((), minval=0, maxval=num_actions, dtype=tf.int32)

    return tf.argmax(tf.reshape(model(tf.reshape(state, [1, -1])), [-1]), output_type=tf.int32)


@tf.function
def train_model(model: keras.Model, target_model: keras.Model, optimizer: keras.optimizers.Optimizer, state: tf.Tensor, next_state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, done: tf.Tensor) -> tf.Tensor:
    q_next = target_model(next_state)

    q_action_target = reward + GAMMA * \
        tf.reduce_max(q_next, axis=1) * (1 - tf.cast(done, tf.float32))

    with tf.GradientTape() as tape:
        q_curr = model(state)
        indices = tf.stack([tf.range(BATCH_SIZE), action], axis=1)
        q_curr_action = tf.gather_nd(q_curr, indices)
        loss = tf.reduce_mean((q_curr_action - q_action_target) ** 2)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


class Agent:
    def __init__(self, input_space, output_space, hidden_layers):
        self.input_space = input_space
        self.output_space = output_space
        self.hidden_layers = hidden_layers

        self.model = self.create_new_model()
        self.model_target = self.create_new_model()
        self.target_update_counter = 0

        self.eps = 1.0

        self.replay_bank = ReplayWrapper(BATCH_SIZE, input_space)

        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def create_new_model(self):
        return keras.Sequential([
            keras.layers.Input(self.input_space),
            *[
                keras.layers.Dense(hidden_size, activation='relu') for hidden_size in self.hidden_layers
            ],
            keras.layers.Dense(self.output_space, activation=None)
        ])

    def update_target(self):
        self.model_target.set_weights(self.model.get_weights())

    def choose_action(self, state):
        return int(choose_action_tf(self.model,
                                    tf.constant(self.eps, dtype=tf.float32),
                                    tf.constant(state, dtype=tf.float32),
                                    self.output_space))

    def remember(self, state, new_state, action, reward, done):
        self.replay_bank.write_trajectory(
            state, new_state, action, reward, done)

    def train_step(self):
        if not self.replay_bank.has_enough_data():
            return 0

        batch = self.replay_bank.take_batch()

        self.eps = max(self.eps - EPS_DEC, EPS_MIN)

        self.target_update_counter += 1
        if self.target_update_counter >= TARGET_UPDATE_INTERVAL:
            self.target_update_counter = 0
            self.update_target()

        return float(train_model(self.model, self.model_target, self.optimizer, batch['state'], batch['next_state'],
                                 batch['action'], batch['reward'], batch['done']))
