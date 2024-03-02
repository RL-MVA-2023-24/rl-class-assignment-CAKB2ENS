from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import os
import torch

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:   
    def __init__(self):
        self.model = []
        self.device = 'cpu'
    
    def act(self, observation, use_random=False):
        return self.greedy_action(observation)

    def save(self, path):
        pass

    def load(self):
        # import os
        # import torch
        self.model = torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'), map_location=self.device)

    def greedy_action(self, observation):
        # import torch
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to('cpu'))
            return torch.argmax(Q).item()



