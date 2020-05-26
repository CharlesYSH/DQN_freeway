#!/usr/bin/python3
#coding=utf-8

import os
import argparse
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device: %s"%device)

from dqn import DQN
from agent import RandomAgent
from environment import Atari

parser = argparse.ArgumentParser()
parser.add_argument("--train_ep", default=800, type=int)
parser.add_argument("--mem_capacity", default=80000, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.00025, type=float)
parser.add_argument("--gamma", default=0.999, type=float)
parser.add_argument("--epsilon_start", default=1.0, type=float)
parser.add_argument("--epsilon_final", default=0.1, type=float)
parser.add_argument("--epsilon_decay", default=1000000, type=float)
parser.add_argument("--target_step", default=10000, type=int)
parser.add_argument("--eval_per_ep", default=10, type=int)
parser.add_argument("--save_per_ep", default=50, type=int)
parser.add_argument("--save_dir", default="./model")
parser.add_argument("--log_file", default="./log.txt")
parser.add_argument("--load_model", default=None)

def train():
    num_episodes = args.train_ep
    save_model_per_ep = args.save_per_ep
    log_fd = open(args.log_file,'w')

    agent = DQN(args)
    env = Atari()

    if args.load_model:
      agent.restore_model(args.load_model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    global_steps = 0
    for i_episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()

        for _ in range(10000):
            #env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            
            if done:
                next_state = None

            agent.memory.push(  state, \
                                action, \
                                next_state, \
                                torch.tensor([[reward]], device=device), \
                                torch.tensor([done], device=device, dtype=torch.bool))
            state = next_state
            episode_reward += reward
            global_steps += 1 

            if global_steps > 50000:
                agent.update()

            if done:
                train_info_str = "Episode: %6d, interaction_steps: %6d, reward: %2d, epsilon: %f"%(i_episode, agent.interaction_steps, episode_reward, agent.epsilon)
                print(train_info_str)
                log_fd.write(train_info_str)
                break

        if i_episode % save_model_per_ep == 0:
            agent.save_model(args.save_dir)
            print("[Info] Save model at '%s' !"%args.save_dir)

        if i_episode % args.eval_per_ep == 0:
            test_env = Atari()
            episode_reward = 0
            state = test_env.reset()
            for _ in range(10000):
                action = agent.evaluate_action(state)
                state, reward, done, _ = test_env.step(action.item())
                episode_reward += reward
            eval_info_str = "Evaluation: True, Episode: %6d, Interaction_steps: %6d, evaluate reward: %2d"%(i_episode, agent.interaction_steps, episode_reward)
            print(eval_info_str)
            log_fd.write(eval_info_str)
    
    log_fd.close()


def test_random():
    agent = RandomAgent()
    env = Atari()

    for i_episode in range(10):
        episode_reward = 0
        state = env.reset()

        for _ in range(10000):
            env.env.render()
            action = agent.evaluate_action(state)
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            episode_reward += reward

            if done:
                print("Episode: %6d, interaction_steps: %6d, reward: %2d, epsilon: %f"%(i_episode, agent.interaction_steps, episode_reward, 1.0))
                break


def test(path):
    epsilon = args.epsilon_final
    agent = DQN(args)
    env = Atari()
        
    agent.restore_model(path)

    for i_episode in range(10):
        episode_reward = 0
        state = env.reset()

        for _ in range(10000):
            env.env.render()
            action = agent.evaluate_action(state, epsilon)
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            episode_reward += reward

            if done:
                print("Episode: %6d, interaction_steps: %6d, reward: %2d, epsilon: %f"%(i_episode, agent.interaction_steps, episode_reward, epsilon))
                break


if __name__ == '__main__':
    args = parser.parse_args()
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        print("[Info] make directory '%s'"%args.save_dir)

    print("Ramdom Agent")
    test_random()
    print("Train DQN")
    train()
    print("DQN Agent")
    test("./source_model/q_target_checkpoint_1947648.pth")



