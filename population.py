import torch
import random

from prime_model import PrimeModel

# 20 0 or 1 bits to represent the number
inputs = 20
# 2 possible outputs: prime or not prime
outputs = 2

device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")


class Population:
    def __init__(self) -> None:
        self.num_agents = 25
        self.agents = []
        self.best_agent = None

        self.mutation_rate = 0.1

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
        self.agents = self.new_agents

    def find_best_agent(self, x, y):
        min_loss = float('inf')
        for agent in self.agents:
            loss = agent.get_loss(x, y)
            if loss < min_loss:
                min_loss = loss
                self.best_agent = agent
        return self.best_agent

    def simulate(self, x, y, num_generations=100):
        for i in range(num_generations):
            self.initialize_population(inputs, outputs)
            self.find_best_agent(x, y)
            print(f"Generation {i} loss: {self.best_agent.loss}")


def gen_primes(n):
    """ Returns a list of primes < n """
    sieve = [True] * n
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i]:
            sieve[i * i::2 * i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return sieve


upper_bound = 2 ** (inputs - 2) - 1


def main():
    primes = gen_primes(upper_bound)

    x = torch.tensor([list(map(int, [*format(i, "020b")]))
                     for i in range(upper_bound)], dtype=torch.float).to(device)
    print(x.shape)
    y = torch.tensor([[0, 1] if prime else [1, 0]
                     for prime in primes], dtype=torch.float).to(device)
    print(y.shape)

    population = Population()
    population.simulate(x, y)


if __name__ == "__main__":
    main()
