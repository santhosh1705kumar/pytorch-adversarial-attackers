import torch

def evaluate(model, image, label):
    """Evaluates the model's accuracy on adversarial examples."""
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(label.view_as(pred)).sum().item() / label.size(0)
    print(f"Accuracy on adversarial example: {accuracy * 100:.2f}%")
    return accuracy
