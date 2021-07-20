import torch
import torchvision
from torchvision import transforms
from PIL import Image

print(f"All model available - {dir(torchvision.models)}")

# Load the ResNet Model
resnet = torchvision.models.resnet101(pretrained=True)
# Put the network in eval model, If we forget to do that, some pretrained models, like batch normalization and dropout,
# will not produce meaningful answers, just because of the way they work internally.
# Now that eval has been set, we’re ready for inference:

resnet.eval()

# Create a preprocessor
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)  # Predefined values as used in training of the Resnet model

img = Image.open("./data/dog_1.jpg")
img_t = preprocess(img=img)

batch_t = torch.unsqueeze(img_t, 0)

out = resnet(batch_t)

print(out)

# Now read the data classes
with open("./data/imagenet_classes.txt", "rt") as fp:
    labels = [line.strip() for line in fp.readlines()]

# Find the highest value's index
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(
    f"Predicted class = {labels[index[0]]}, Confidence = {percentage[index[0]].item()} %"
)
