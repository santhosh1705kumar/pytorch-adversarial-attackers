def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize image like ImageNet dataset."""
    return (image - mean) / std
