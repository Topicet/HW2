import gymnasium as gym
import numpy as np
import cv2
import os
import torch

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.encoding import poisson
from bindsnet.learning import MSTDPET
from bindsnet.network.nodes import Nodes

import matplotlib.pyplot as plt


class spaceInvaders():

    def __init__(self):
        print("Initializing Space Invaders environment...")
        self.env = gym.make("SpaceInvaders-v0", render_mode='human')
        self.state = self.env.reset()[0]
        print("Initial state shape:", self.state.shape)

        self.terminated = False
        self.truncated = False
        self.reward = 0

        self.spike_history = []
        self.action_history = []
        self.reward_history = []
        self.input_spike_counts = []
        self.output_currents = []


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        print("Setting up spiking neural network...")
        self.network = Network()
        self.input_layer = Input(n=84*84, shape=(84, 84))        
        class ReLUNodes(Nodes):
            def forward(self, x):
                self.s = torch.relu(x)
                return self.s

        self.output_layer = ReLUNodes(n=6)  # 6 actions in SpaceInvaders

        self.network.add_layer(self.input_layer, name="Input")
        self.network.add_layer(self.output_layer, name="Output")
        self.output_layer.v_thresh = torch.tensor(0.01)
        self.network.add_connection(Connection(
            source=self.input_layer,
            target=self.output_layer,
            w=torch.rand(self.input_layer.n, self.output_layer.n),  # random weights
        ), source='Input', target='Output')

        with torch.no_grad():
            self.network.connections[("Input", "Output")].w.uniform_(0, 1)
        print("Network initialized with random weights.")

    def preprocess_frame(self, obs):
        # print("Preprocessing frame...")
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        # print("Frame preprocessed: shape =", resized.shape)
        return resized / 255.0  # Normalizing to [0,1]

    def encode_frame(self, frame, time=100):
        # print("Encoding frame into spikes...")
        
        flattened = frame.flatten().astype(np.float32)
        pixel_vals = np.clip(flattened * 10.0, 0.0, 1.0)  # brightness boost

        # Use a manual Poisson generator
        rand_vals = np.random.rand(time, len(pixel_vals))
        spikes = (rand_vals < pixel_vals).astype(np.float32)

        spikes_tensor = torch.tensor(spikes, dtype=torch.float32, device=self.device)
        
        # print("Total input spikes this frame:", spikes_tensor.sum().item())
        return spikes_tensor



    def run(self):
        print("Starting game loop...")
        while not self.truncated and not self.terminated:
            # print("New frame processing...")
            frame = self.preprocess_frame(self.state)
            spikes = self.encode_frame(frame)

            # print("Resetting network state variables...")
            self.network.reset_state_variables()

            inputs = {'Input': spikes}
            # print("Inputs prepared for network:", inputs['Input'].shape)

            # Run the network with current frame and reward
            # print("Running network...")

            w = self.network.connections[("Input", "Output")].w
            inp_spikes = inputs["Input"]  # shape [100, 7056]
            mean_input = inp_spikes.mean(dim=0)  # average firing rate per neuron
            current = torch.matmul(mean_input, w)  # shape [6]
            # print("Expected output current:", current)
            self.network.run(inputs=inputs, time=spikes.shape[0], reward=self.reward)

            # Choose action based on output spike counts *after* running the network
            spike_counts = self.network.layers['Output'].s.sum(dim=0)



            # print("Spike counts from output layer:", spike_counts)

            action = torch.argmax(spike_counts).item()
            # print(f"Selected action: {action}")

            self.spike_history.append(spike_counts.cpu().numpy())
            self.action_history.append(action)
            self.reward_history.append(self.reward)
            self.input_spike_counts.append(spikes.sum().item())
            self.output_currents.append(current.cpu().numpy())
            # Take action in the game → get new state and reward
            # print("Taking action in the environment...")
            self.state, self.reward, self.terminated, self.truncated, _ = self.env.step(action)
            # print(f"New state received. Reward: {self.reward}, Terminated: {self.terminated}, Truncated: {self.truncated}")

        print("Game loop ended. Closing environment...")
        self.env.close()
        self.plot_stats()


    def print_info(self):
        print(f"State: {self.state}\nReward: {self.reward}\nTerminated: {self.terminated}\nTruncated: {self.truncated}")

    import os

    def plot_stats(self):
        spikes = np.array(self.spike_history)
        currents = np.array(self.output_currents)

        os.makedirs("plots", exist_ok=True)  # save to a folder

        # Output Neuron Activations
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title("Output Neuron Activations Over Time")
        for i in range(spikes.shape[1]):
            plt.plot([s[i] for s in spikes], label=f"Neuron {i}")
        plt.legend()

        # Output Currents
        plt.subplot(2, 1, 2)
        plt.title("Output Currents Over Time")
        for i in range(currents.shape[1]):
            plt.plot([c[i] for c in currents], label=f"Neuron {i}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/output_neurons_and_currents.png")
        plt.close()

        # Action History
        plt.figure()
        plt.plot(self.action_history)
        plt.title("Action Taken Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Action")
        plt.savefig("plots/action_history.png")
        plt.close()

        # Input Spike Count
        plt.figure()
        plt.plot(self.input_spike_counts)
        plt.title("Total Input Spikes Per Frame")
        plt.xlabel("Frame")
        plt.ylabel("Spike Count")
        plt.savefig("plots/input_spikes.png")
        plt.close()

        # Cumulative Reward
        plt.figure()
        plt.plot(np.cumsum(self.reward_history))
        plt.title("Cumulative Reward")
        plt.xlabel("Frame")
        plt.ylabel("Total Reward")
        plt.savefig("plots/cumulative_reward.png")
        plt.close()




if __name__ == '__main__':
    print("Creating Space Invaders instance...")
    my_space_invader = spaceInvaders()
    print("Starting the game...")
    my_space_invader.run()
    print("Game finished.")
