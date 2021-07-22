import torch
import h5py

# Example 1
# So you want to convert to greyscale, you get the typical weights for RGB from some source as
raw_weights = torch.tensor([0.2126, 0.7152, 0.0722])

# Now you've following scenarios to make greyscale
img_t = torch.randn(3, 5, 5)  # channel x rows x columns
batch_t = torch.randn(10, 3, 5, 5)  # batch x channel x rows x columns
mean_across_all_colors_img = img_t.mean(
    -3
)  # 3rd from last is always color channel, that's the trick!
mean_across_all_colors_batch = img_t.mean(-3)

# Gotta unsqueeze the weights to be able to multiply the tensors
weights = raw_weights.unsqueeze(-1).unsqueeze_(
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

# THere's also a einsum technique for naming dimensions
img_gray_weighted_fancy = torch.einsum(
    "...chw,c->...hw", img_t, raw_weights
)  # That raw weights have to be multiplied by right dimension is mentioned by using c
batch_gray_weighted_fancy = torch.einsum("...chw,c->...hw", batch_t, raw_weights)
print(f"Fancy shape after using einsum = {batch_gray_weighted_fancy.shape}")

# Named Tensors - Starting pytorch 1.3, we can do
weights_named = torch.tensor([5, 1, 3], names=["channels"])
# Let's add names to above image stuff
img_named = img_t.refine_names(
    ..., "channels", "rows", "columns"
)  # ... is ellipsis and it means all other dimentions
batch_named = batch_t.refine_names(..., "channels", "rows", "columns")
print(batch_named.shape)
print(batch_named.names)

# For operations with two inputs, in addition to the usual dimension checks—whether sizes are the same, or if one is
# 1 and can be broadcast to the other—PyTorch will now check the names for us. So far, it does not automatically
# align dimensions, so we need to do this explicitly. The method align_as returns a tensor with missing dimensions
# added and existing ones permuted to the right order
print(f"Named Weights before aligning with Named Image = {weights_named}")
weights_named_aligned = weights_named.align_as(img_named)
print(f"Named Weights after aligning with Named Image = {weights_named_aligned}")
print(weights_named_aligned[1, 0, 0])

# Now, with names, operations become clearer
grey_named = (weights_named_aligned * img_named).sum("channels")
print(grey_named.shape)
print(grey_named.names)
print(grey_named)
print(img_named[..., :3].shape)
print(img_named[..., :3].names)
print(weights_named_aligned.names)
print((img_named[..., :3] * weights_named_aligned).shape)
print((img_named[..., :3] * weights_named_aligned).sum("channels"))
# gray_named = (img_named[..., :3] * weights_named).sum('channels') will not run as it has not been aligned
print(weights_named.names)
print(img_named.names)
# Finally, we can drop the names -
img_unnamed = img_named.rename(None)

# Part 2 Storage and other things
# Tensors are Views on torch.Storage instances which stores the data in contiguous memory chunks.
# A Tensor instance is a View which indexes into this Storage using offset and per-dimension strides
points = torch.tensor([[2, 3], [5, 3], [45, 1], [-12.90, 90]])
print(f"Storage = {points.storage()}")
print(f"Offsets = {points.storage_offset()}")
# Even though the tensor reports itself as having three rows and two columns, the storage under the hood is a
# contiguous array of size 6. In this sense, the tensor just knows how to translate a pair of indices into a location
# in the storage.
print(f"Access storage directly - {points.storage()[0]}")
# You can also modify the storage in-place
points.storage()[0] = 909
print(f"Post Storage modification - {points[0, 0]}")
# Some operations modify tensors in place and have "_" at the end
points.zero_()
print(f"Post in-place modification - {points}")

# All well and good .. what does stride and offset mean though? say we have a 2D tensor, A, it will have a scalar (
# it's always scalar) offset and a 1D stride (always 1D) of length 2 (= number of dimensions). then A[i,
# j] = (storage of A) [offset + stride[0]*i + stride[1]*j]
points = torch.tensor([[2, 3], [5, 3], [45, 1], [-12.90, 90]])
i, j = 3, 1
print(f"This should be True")
print(f"Tensor = {points}")
print(f"Stride = {points.stride()}")
print(f"Offset = {points.storage_offset()}")
print(f"Storage = {points.storage()}")
print(
    points[i, j]
    == points.storage()[
        points.storage_offset() + points.stride()[0] * i + points.stride()[1] * j
    ]
)

# Doing a subset selection on Tensor gives another View on the underlying storage, modifying it then will lead to
# changing of original Tensor, if you don't want this, close the Tensor and this will assign new memory
print(f"Before changing the selected Tensor = {points[0, 0]}")
points_sub = points[:, 0]
points_sub[0] = 101
print(f"After changing the selected Tensor = {points[0, 0]}")
print(f"Instead modifying a clone now")
points_sub = points[:, 0].clone()
points_sub[0] = 10001
print(f"After changing the selected cloned Tensor = {points[0, 0]}")

# Interesting thing - in this framework, calculating transpose is just a matter of rearranging strides!
a_tensor = torch.randn([3, 2, 2])
print(f"Storage of original = {a_tensor.storage()}")
print(f"Shape of original = {a_tensor.shape}")
print(f"ID of storage of original = {id(a_tensor.storage())}")
print(f"Stride of original = {a_tensor.stride()}")
print(f"Let's create a transpose by interchanging dimensions 0 and 2")
a_tensor_t = a_tensor.transpose(0, 2)
print(f"Shape of transposed= {a_tensor_t.shape}")
print(f"ID of storage of transposed = {id(a_tensor_t.storage())}")
print(f"Stride of transposed = {a_tensor_t.stride()}")
# A tensor whose values are laid out in the storage starting from the rightmost dimension onward (that is,
# moving along rows for a 2D tensor) is defined as contiguous. Contiguous tensors are convenient because we can visit
# them efficiently in order without jumping around in the storage (improving data locality improves performance
# because of the way memory access works on modern CPUs). This advantage of course depends on the way algorithms visit
# Some tensor operations in PyTorch only work on contiguous tensors, such as view. In that case, PyTorch will throw
# an informative exception and require us to call contiguous explicitly. It’s worth noting that calling contiguous
# will do nothing (and will not hurt performance) if the tensor is already contiguous.
print(f"Is original contiguous? - {a_tensor.is_contiguous()}")
print(f"Is transposed contiguous? - {a_tensor_t.is_contiguous()}")

# As contiguous tensors are good to have (even mandatory for some Ops), let's make the transposed one contiguous We
# can obtain a new contiguous tensor from a non-contiguous one using the contiguous method. The content of the tensor
# will be the same, but the stride will change, as will the storage
a_tensor_t = a_tensor_t.contiguous()
print(
    f"ID of storage after making the transposed one contiguous = {id(a_tensor_t.storage())}"
)
print(f"Seems the ID is same, is the content of Storage changed then?")
print(f"Storage of transposed and contiguous = {a_tensor_t.storage()}")
print(f"Aha! the storage is indeed shuffled inplace!")

# GPUs don't work on my machine yet so following won't work, but ewhere available, this stores the tensor on GPU instead
# points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')

# # You can convert tensors to numpy and back, but take care that the dtype is correct as tensors use float32 as
# default whereas numpy uses float64 as default -
a_tensor_np = a_tensor.numpy()
print(f"Dtype of original = {a_tensor.dtype}")
print(f"Dtype of numpy array = {a_tensor_np.dtype}")
a_tensor_np_to_tensor = torch.from_numpy(a_tensor_np)
print(f"Dtype of Tensor converted from numpy to back = {a_tensor_np_to_tensor.dtype}")

# You provide the API contract and have your own Tensor! Say you call torch.add(a, b) where and b are tensors.
# PyTorch will cause the right computation functions to be called regardless of whether our tensor is on the CPU or
# the GPU or what it's dtype is etc. This is accomplished through a dispatching mechanism, and that mechanism can
# cater to other tensor types by hooking up the user-facing API to the right backend functions. Sure enough,
# there are other kinds of tensors: some are specific to certain classes of hardware devices (like Google TPUs),
# and others have data-representation strategies that differ from the dense array style we’ve seen so far. For
# example, sparse tensors store only nonzero entries, along with index information. The PyTorch dispatcher claims to
# be extensible; the subsequent switching done to accommodate the various numeric types is a fixed aspect of the
# implementation coded into each backend. There are other types like quantized tensors which are implemented as
# another type of tensor with a specialized computational backend. Sometimes the usual tensors we use (so far in this
# notebook) are called dense or strided to differentiate them from tensors using other memory layouts.

# Saving tensors on Disk is a matter of calling torch.save
torch.save(points, "./data/saved_points_tensor.t")
# And loading
point_loaded = torch.load("./data/saved_points_tensor.t")
print(points.dtype == point_loaded.dtype)


# Finally, we can use HDF5 framework to save Tensors as general multidimensional arrays by converting them to numpy
# arrays


f = h5py.File("./data/saved_points_h5.hdf5", "w")
dset = f.create_dataset("coords", data=points.numpy())
f.close()
# Can also read only some bits from the disk without loading rest in memory
f_r = h5py.File("./data/saved_points_h5.hdf5", "r")
dset_r = f_r["coords"]  # It's not yet loaded in memory

last_points = torch.from_numpy(
    dset_r[:-2]
)  # Note that what h5py returns is not exactly numpy array but like a numpy
# array so we can still convert to tensor
f_r.close()  # If we close before getting the dset_r, we're obviously gonna get an error
print(last_points)
