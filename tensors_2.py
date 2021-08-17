"""
This module shows how to represent various types of datasets as Tensors
"""
import os
import pandas as pd
import numpy as np
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


#########
# Volumetric Images - CT Scans etc.
#########

# There’s no fundamental difference between a tensor storing volumetric data versus image data. We just have an extra
# dimension, depth, after the channel dimension, leading to a 5D tensor of shape N × C × D × H × W. Let’s load a
# sample CT scan using the volread function in the imageio module, which takes a directory as an argument and
# assembles all Digital Imaging and Communications in Medicine (DICOM) files2 in a series in a NumPy 3D array

vol_arr = torch.from_numpy(
    imageio.volread(r"./data/volumetric-dicom/2-LUNG 3.0  B70f-04083", format="DICOM")
)
print(vol_arr.shape)
# Now , we gotta make it conform Channel x depth x height x width
# add a channel
vol_arr = torch.unsqueeze(vol_arr, dim=0)  # Insert one more dimension at 0th position
print(vol_arr.shape)

# That's it!

#########
# Tabular Data
#########
df = pd.read_csv("./data/tabular-wine/winequality-white.csv", sep=";", dtype=np.float32)
wineeq = torch.from_numpy(df.to_numpy())
print(wineeq.shape, wineeq.dtype)

data = wineeq[:, :-1]
target = wineeq[:, -1].long()

print(data.shape, target.shape)
# print(target.unsqueeze(1).shape)
# Observe there are 10 values in target, we can either use them as a score or encode one hot vectors instead
print(target)
onehot_target = torch.zeros(target.shape[0], 10)
onehot_target.scatter_(dim=1, index=target.unsqueeze(1), value=1)
print(onehot_target)
print(torch.mean(onehot_target, dim=0))
data_mean = torch.mean(data, dim=0)
data_sd = torch.std(data, dim=0)
data_normalized = (data - data_mean) / data_sd
bad_data = data[target <= 3]
okay_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]
bad_mean = torch.mean(bad_data, dim=0)
okay_mean = torch.mean(okay_data, dim=0)
good_mean = torch.mean(good_data, dim=0)
for col, b, o, g in zip(df.columns, bad_mean, okay_mean, good_mean):
    print(f"{col} {b} {o} {g}")
# Seems like higher total sulphur dioxide is the problem, let's manually build a quick model
so2_threshold = 141.83
predicted_indexes = data[:, 6] <= so2_threshold
print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())
actual_indices = target > 5
print(actual_indices.sum())
hits = torch.sum(actual_indices & predicted_indexes).item()
total_predicted = torch.sum(predicted_indexes).item()
total_true = torch.sum(actual_indices).item()
print(hits, total_predicted, total_true)
print(f"Precision = {hits / total_predicted}\nRecall={hits / total_true}")

##############
# Time Series
##############

# In the source data, each row is a separate hour of data (figure 4.5 shows a transposed version of this to better
# fit on the printed page). We want to change the row-per-hour organization so that we have one axis that increases
# at a rate of one day per index increment, and another axis that represents the hour of the day (independent of the
# date). The third axis will be our different columns of data (weather, temperature, and so on).


# TODO below code gives wrong number of channels - fix this df = pd.read_csv(
#  './data/bike-sharing-dataset/hour-fixed.csv', dtype=np.float32, skiprows=0, converters={1: lambda x: float(x[
#  8:10])}) bikes = torch.from_numpy(df.values) print(df.head()) print(bikes.shape, bikes.stride())

# Below works fine though
bikes_numpy = np.loadtxt(
    "./data/bike-sharing-dataset/hour-fixed.csv",
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    converters={1: lambda x: float(x[8:10])},
)
bikes = torch.from_numpy(bikes_numpy)
print(bikes.shape, bikes.stride())
daily_bikes = bikes.view(-1, 24, bikes.shape[1])  # Might as well be 730, 24, 17
print(daily_bikes.shape, daily_bikes.stride())

# Where we want to go? - N x C x L
# What we have? - N sequences of L hours in a day, for C channels
# Gotta transpose this, boss
daily_bikes = daily_bikes.transpose(1, 2)  # interchange 1 and 2 dimension


####################
# Text Data
####################
with open("./data/jane-austen/1342-0.txt", encoding="utf8") as f:
    text = f.read()
lines = text.split("\n")
line = lines[200]
print(line)
letter_t = torch.zeros(
    len(line), 128
)  # 128 because we only care about ASCII characters, 256 for UTF-8

for i, letter in enumerate(line.lower().strip()):
    letter_index = (
        ord(letter) if ord(letter) < 128 else 0
    )  # Return the Unicode code point for a one-character string
    letter_t[i][letter_index] = 1

# That's it! We've encoded one sentence's characters
print(letter_t)


def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace("\n", " ").split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


words_in_line = clean_words(line)
print(line, words_in_line)

word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}
print(len(word2index_dict), word2index_dict["impossible"])

word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print("{:2} {:4} {}".format(i, word_index, word))
print(word_t.shape)
