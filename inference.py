import torch
import torch.nn as nn
from model import CIFAR100Model

from PIL import Image
import torchvision.transforms as transforms

classes = [
    # aquatic mammals
    "beaver", "dolphin", "otter", "seal", "whale",

    # fish
    "aquarium fish", "flatfish", "ray", "shark", "trout",

    # flowers
    "orchids", "poppies", "roses", "sunflowers", "tulips",

    # food containers
    "bottles", "bowls", "cans", "cups", "plates",

    # fruit and vegetables
    "apples", "mushrooms", "oranges", "pears", "sweet peppers",

    # household electrical devices
    "clock", "computer keyboard", "lamp", "telephone", "television",

    # household furniture
    "bed", "chair", "couch", "table", "wardrobe",

    # insects
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",

    # large carnivores
    "bear", "leopard", "lion", "tiger", "wolf",

    # large man-made outdoor things
    "bridge", "castle", "house", "road", "skyscraper",

    # large natural outdoor scenes
    "cloud", "forest", "mountain", "plain", "sea",

    # large omnivores and herbivores
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",

    # medium-sized mammals
    "fox", "porcupine", "possum", "raccoon", "skunk",

    # non-insect invertebrates
    "crab", "lobster", "snail", "spider", "worm",

    # people
    "baby", "boy", "girl", "man", "woman",

    # reptiles
    "crocodile", "dinosaur", "lizard", "snake", "turtle",

    # small mammals
    "hamster", "mouse", "rabbit", "shrew", "squirrel",

    # trees
    "maple", "oak", "palm", "pine", "willow",

    # vehicles 1
    "bicycle", "bus", "motorcycle", "pickup truck", "train",

    # vehicles 2
    "lawn-mower", "rocket", "streetcar", "tank", "tractor"
]

model = CIFAR100Model()
model.load_state_dict(torch.load("cifar100_model.pth", map_location='cpu'))
model.eval()

transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408,), (0.2675, 0.2565, 0.2761,))])

image = Image.open("test_images/apple.avif").convert("RGB")
image = transforms(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1)

print("Class:", classes[prediction.item()])