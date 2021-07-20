import torch

# Example 1
# So you want to convert to greyscale, you get the typical weights for RGB from some source as
weights = torch.tensor([0.2126, 0.7152, 0.0722])

# Now you've following scenarios to make greyscale
img_t = torch.randn(3, 5, 5)  # channel x rows x columns
batch_t = torch.randn(10, 3, 5, 5)  # batch x channel x rows x columns
mean_across_all_colors_img = img_t.mean(
    -3
)  # 3rd from last is always color channel, that's the trick!
mean_across_all_colors_batch = img_t.mean(-3)

# Gotta unsqueeze the weights to be able to multiply the tensors
weights = weights.unsqueeze(-1).unsqueeze_(
    -1
)  # Why unsqueeze_ 2nd time? This modifies in-place,
# thus reusing the tensor from first operation (https://github.com/deep-learning-with-pytorch/dlwpt-code/issues/36)

print(f"Shape of colo weights after unsqueezing = {weights.shape}")
img_weights = img_t * weights
batch_weights = (
    batch_t * weights
)  # why does this work? because by default 1s are inserted at the beginning of
# tensor if shape does not match (broadcasting)
# Now let's add the normalize colors to give greyscale
img_weights_grey = img_weights.sum(-3)  # -3 is the color channel, remember?
batch_weights_grey = batch_weights.sum(-3)

print(batch_weights_grey.shape)
