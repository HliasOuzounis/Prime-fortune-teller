import torch

device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")

class PrimeModel(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=32):
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.hidden_size = hidden_size
        super(PrimeModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_outputs),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1),
        )
        
        self.loss = None

        self.score = None
        self.to(device)

    def forward(self, x):
        return self.model(x)
    
    def get_loss(self, x, y):
        predictions = self.forward(x)
        self.loss = torch.nn.functional.cross_entropy(predictions, y)
        return self.loss
    
    def marry(self, other, mutation_rate=0.05):
        child = PrimeModel(self.inputs, self.outputs, self.hidden_size)

        for child_param, self_param, other_param in zip(child.parameters(), self.parameters(), other.parameters()):
            child_param.data.copy_((self_param.data + other_param.data) / 2 + torch.randn(self_param.size()).to(device) * mutation_rate)
        return child