import tensorflow as tf
import reverb

REPLAY_BUFFER_MAX_SIZE = 100000


class ReplayWrapper:
    def __init__(self, batch_size, state_dim):
        self.batch_size = batch_size

        self.srv = reverb.Server(
            tables=[
                reverb.Table(
                    name='training_table',
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=REPLAY_BUFFER_MAX_SIZE,
                    rate_limiter=reverb.rate_limiters.MinSize(batch_size),
                    signature={
                        'state': tf.TensorSpec((state_dim), dtype=tf.float32),
                        'next_state': tf.TensorSpec((state_dim), dtype=tf.float32),
                        'action': tf.TensorSpec((), dtype=tf.int32),
                        'reward': tf.TensorSpec((), dtype=tf.float32),
                        'done': tf.TensorSpec((), dtype=tf.bool)
                    }
                )
            ]
        )

        self.client = reverb.Client(f'localhost:{self.srv.port}')

        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f'localhost:{self.srv.port}',
            table='training_table',
            max_in_flight_samples_per_worker=10)

        self.dataset = dataset.batch(batch_size)

        self.enough_data = False

    def write_trajectory(self, state, next_state, action, reward, done):
        with self.client.trajectory_writer(num_keep_alive_refs=1) as writer:
            writer.append({
                'state': state,
                'next_state': next_state,
                'action': tf.cast(action, tf.int32),
                'reward': tf.cast(reward, tf.float32),
                'done': done
            })

            writer.create_item(table='training_table', priority=1, trajectory={
                'state': writer.history['state'][-1],
                'next_state': writer.history['next_state'][-1],
                'action': writer.history['action'][-1],
                'reward': writer.history['reward'][-1],
                'done': writer.history['done'][-1]
            })

    def has_enough_data(self) -> bool:
        if self.enough_data:
            return True
        if self.client.server_info()['training_table'].current_size >= self.batch_size:
            self.enough_data = True
        return self.enough_data

    def take_batch(self):
        for b in self.dataset.take(1):
            return b.data
