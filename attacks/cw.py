import torch
import torch.optim as optim
import torch.nn.functional as F

def CW(model, image, label, c=1, kappa=0, steps=1000, lr=0.01):
    perturbed_image = image.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([perturbed_image], lr=lr)
    for _ in range(steps):
        output = model(perturbed_image)
        one_hot_label = torch.nn.functional.one_hot(label, num_classes=output.shape[1]).float()
        loss = torch.sum((output * (1 - one_hot_label)) - (output * one_hot_label) + kappa)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.clamp(perturbed_image, 0, 1)
