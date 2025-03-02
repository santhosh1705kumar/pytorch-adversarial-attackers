import torch
import torch.nn.functional as F

def PGD(model, image, label, epsilon=0.03, alpha=0.005, iterations=10):
    perturbed_image = image.clone().detach().requires_grad_(True)
    for _ in range(iterations):
        output = model(perturbed_image)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
    return torch.clamp(perturbed_image, 0, 1)
