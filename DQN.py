
import numpy as np
import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# setting up cart pole env
env = gym.make("CartPole-v1", render_mode="human")
env.reset()

# Global variables
replay_buffer = deque(maxlen = 3000)
batch_size = 100
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = tf.keras.losses.mean_squared_error

input = env.observation_space.shape
n_output = env.action_space.n

def sample_experience(batch_size):
    indices = np.random.randint(len(replay_buffer), size= batch_size)
    batch = [replay_buffer[index] for index in indices]
    array = [np.array([experience[field_index] for experience in batch]) #  list comprehension
            for field_index in range(6)]
    return array

def play_one_step(state, epsilon):
    action = epsilon_greedy(state, epsilon)
    next_state, reward, done, truncated, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done, truncated))
    return next_state, reward, done, truncated, info

def epsilon_greedy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_output)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)  # 2d array
        return Q_values.argmax()

# Q-network
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(units=64, activation="elu", input_shape=input,),
     tf.keras.layers.Dense(units=64, activation="elu"),
     tf.keras.layers.Dense(n_output, activation="linear")])
# Training
def training_step(batch_size):
    experience = sample_experience(batch_size)
    states, actions, rewards, next_states, dones, truncateds = experience
    next_Q_values = model.predict(next_states, verbose=0)
    max_next_Q_values = next_Q_values.max(axis = 1)
    runs = 1.0 - (dones | truncateds)
    target_Q_values = rewards + runs *  discount_factor* max_next_Q_values
    print(f"Target_Q {target_Q_values}")
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_output)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def play():
    rewards = []
    episodes = []
    for episode in range(600):
        total_reward = 0 # for metrics
        obs, info = env.reset()
        print(f"Episode #{episode}")
        for step in range(200):
            epsilon = max(1-episode / 500, 0.01)
            print(f"Epsilon {epsilon}")
            obs, reward, done, truncated, info = play_one_step(obs, epsilon)
            total_reward += reward
            if done or truncated:
                rewards.append(total_reward)
                episodes.append(episode)
                break
        if episode > 50:
            print(f"Training begins")
            training_step(batch_size)

    return rewards, episodes


y, x= play()

plt.plot(x, y)
plt.show()