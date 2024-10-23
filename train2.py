import numpy as np
import torch
from rocket import Rocket
from policy2 import DQN
import matplotlib.pyplot as plt
import utils
import os
import glob
import torch.nn.functional as F

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'  # 'hover' or 'landing'

    max_m_episode = 800000
    max_steps = 800
    buffer_size = 10000
    batch_size = 64
    start_learning = 1000
    update_frequency = 4
    target_network_update_frequency = 100

    env = Rocket(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []
   
    net = DQN(input_dim=env.state_dims, output_dim=env.action_dims).to(device)

   # net = DQN(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    # if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
    #     # load the last ckpt
    #     checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
    #     net.load_state_dict(checkpoint['model_G_state_dict'])
    #     last_episode_id = checkpoint['episode_id']
    #     REWARDS = checkpoint['REWARDS']

    replay_buffer = ReplayBuffer(buffer_size)

    for episode_id in range(last_episode_id, max_m_episode):

        # training loop
        state = env.reset()
        episode_reward = 0

        for step_id in range(max_steps):
            action = net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.add(state, action, reward, next_state, done)

            if episode_id % 100 == 1:
                env.render()

            if len(replay_buffer) > start_learning and step_id % update_frequency == 0:
                batch = replay_buffer.sample(batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch

                batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(device)
                q_values = net(torch.tensor(batch_states, dtype=torch.float32).to(device)).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                next_q_values = net.target_q_network(torch.tensor(batch_next_states, dtype=torch.float32).to(device)).max(1)[0]

                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)
                target_q_values = batch_rewards + (1 - batch_dones) * net.gamma * next_q_values



                loss = F.mse_loss(q_values, target_q_values)
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()

                # Update the target network
                if step_id % target_network_update_frequency == 0:
                    net.update_target_network()

            # Prepare for the next step
            state = next_state
            episode_reward += reward
            if done or step_id == max_steps - 1:
                break

        REWARDS.append(episode_reward)
        print('episode id: %d, episode reward: %.3f' % (episode_id, episode_reward))

        if episode_id % 100 == 1:
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id':episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))

