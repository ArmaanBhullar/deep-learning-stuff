"""
This module shows how to represent various types of datasets as Tensors
"""
import os
import torch
import imageio

#########
# Images
#########


img_arr = imageio.imread("./data/dog_2.jpg")
print(f"Shape = {img_arr.shape}")
# At this point, img_arr is a NumPy array-like object with three dimensions: two spatial dimensions, width and height;
# and a third dimension corresponding to the red, green, and blue channels. Any library that outputs a NumPy array
# will suffice to obtain a PyTorch tensor. The only thing to watch out for is the layout of the dimensions. PyTorch
# modules dealing with image data require tensors to be laid out as C × H × W: channels, height, and width,
# respectively, so, we gotta permute the channels of this image!

img = torch.from_numpy(img_arr)
print(f"Shape before permute = {img.shape}")
out = img.permute(2, 0, 1)  # make it C x height x width from height x width x c
print(f"Shape after permute = {out.shape}")
# note that this operation does not make a copy of the tensor data. Instead, out uses the same underlying storage as
# img and only plays with the size and stride information at the tensor level.

# Now let's read images from a directory all into a pre constructed tensor with batch size as first dimension
filenames = [file for file in os.listdir("./data/image-cats") if ".png" in file]
batch_size = len(filenames)
batch = torch.zeros(batch_size, 3, 256, 256)  # We already know the shape of images

for index, file in enumerate(filenames):
    img_arr = imageio.imread(os.path.join("./data/image-cats", file))
    img = torch.from_numpy(img_arr)
    out = img.permute(2, 0, 1)
    out = out[
        :3
    ]  # HEre we mean to keep only the first 3 channels as some images contain extra channels!
    print(out.shape)
    assert out.shape == torch.Size([3, 256, 256]), (
        f"Image not read correctly and/or has unexpected shape of {out.shape}, required 3 color channels and"
        f" 256 pixels by 256 pixels"
    )
    batch[index, ...] = out
print(f"Completed reading all images into Tensor of shape {batch.shape}")

# Normalizing

# Neural networks exhibit the best training performance when the input data ranges roughly from 0 to 1, or from -1 to
# 1 (this is an effect of how their building blocks are defined).

# A typical thing we’ll want to do is cast a tensor to floating-point and normalize the values of the pixels. Casting
# to floating-point is easy, but normalization is trickier, as it depends on what range of the input we decide should
# lie between 0 and 1 (or -1 and 1). One possibility is to just divide the values of the pixels by 255 (the maximum
# representable number in 8-bit unsigned):
print(f"Max in batch  before normalizing = {batch.max()}")

batch = batch.float()
batch = batch / 256
print(f"Max in batch after normalizing = {batch.max()}")

# A second way is to nromalize each channel w.r.t it's mean and std (they have to be computed on training data and
# their values have to be stored to be applied on validation and test sets as well)

for i in range(3):  # Now we've already divided all by 256 but it doesn't matter
    mean = batch[:, i].mean()  # batch[:, i] is equivalent to batch[:, i, ...]
    std = batch[:, i].std()
    batch[:, i] = (batch[:, i] - mean) / std

# There are several other operations on inputs, such as geometric transformations like rotations, scaling,
# and cropping. These may help with training or may be required to make an arbitrary input conform to the input
# requirements of a network, like the size of the image.
