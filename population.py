import torch
import random

from prime_model import PrimeModel, device

# 20 0 or 1 bits to represent the number
inputs = 20
# 2 possible outputs: prime or not prime
outputs = 2

class Population:
    def __init__(self) -> None:
        self.num_agents = 25
        self.agents = []
        self.best_agent = None
        
        self.mutation_rate = 0.05

    def initialize_population(self, inputs, outputs):
        if self.best_agent is None:
            self.agents = [PrimeModel(inputs, outputs)
                           for i in range(self.num_agents)]
            return

        self.new_agents = [self.best_agent]
        for i in range(self.num_agents - 1):
            first_parent = self.best_agent
            second_parent = random.choices(self.agents, [agent.loss for agent in self.agents], k=1)[0]

            child = first_parent.marry(second_parent, self.mutation_rate)
            self.new_agents.append(child)
        for agent in self.agents:
            if not agent == self.best_agent:
                del agent
        self.agents = self.new_agents

    def find_best_agent(self, x, y):
        min_loss = float('inf')
        for agent in self.agents:
            loss = agent.get_loss(x, y)
            if loss < min_loss:
                min_loss = loss
                self.best_agent = agent
        return self.best_agent

    def simulate(self, x, y, num_generations=100, mutation_rate=0.05):
        self.mutation_rate = mutation_rate
        for i in range(num_generations):
            self.initialize_population(inputs, outputs)
            self.find_best_agent(x, y)
            if (i + 1) % 10 == 0:
                print(f"Generation {i + 1} Best loss: {self.best_agent.loss}")
        return self.best_agent
