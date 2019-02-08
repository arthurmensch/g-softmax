import torch
from gsoftmax.continuous import MeasureDistance, DeepLoco

def run_deep_loco():
    x = torch.zeros(2, 1, 64, 64)

    model = DeepLoco()
    potential = MeasureDistance(verbose=True)
    pos, log_weights = model(x)
    f = potential(pos, log_weights)

    target_pos = torch.tensor([[[0.1, 0.2, 0.3]]]).repeat(2, 1, 1)
    target_log_weight = torch.tensor([[[-1.]]]).repeat(2, 1, 1)

    loss = potential.integral(pos, log_weights, target_pos, target_log_weight)
    loss = loss.sum()
    loss.backward()

    pos = pos.detach().numpy()
    weight = torch.exp(log_weights).detach().numpy()

    import matplotlib.pyplot as plt

    plt.scatter(pos[0, :, 0], pos[0, :, 1])
    plt.show()