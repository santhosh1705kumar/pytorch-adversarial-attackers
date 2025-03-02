import torch
import torch.nn.functional as F

def FGSM(model, image, label, epsilon=0.03):
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed_image, 0, 1)
