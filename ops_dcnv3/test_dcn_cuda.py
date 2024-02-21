import sys
sys.path.append("..")
import modules as dcnv3
import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model and move it to the GPU
core_op = getattr(dcnv3, 'DCNv3')
dcn = core_op(
    channels=3,
    kernel_size=(3,1),
    stride=1,
    pad=1,
    dilation=1,
    group=1,
    offset_scale=1.0,
    act_layer='GELU',
    norm_layer='LN',
    dw_kernel_size=None,  # for InternImage-H/G
    center_feature_scale=False,
    strip_conv=1
).to(device)

# Dummy input and move it to the GPU
dummy_input = torch.randn(1, 256, 256, 3).to(device)

# Forward pass
output = dcn(dummy_input)
print(output)

tmp = [param.clone().detach() for param in dcn.parameters()]


# Assuming you have some target tensor
target = torch.randn_like(output).to(device)

# Define a loss function (MSE for demonstration)
criterion = nn.MSELoss()

# Move the model to the GPU
dcn.to(device)

# Compute the loss
loss = criterion(output, target)
# Zero the gradients
dcn.zero_grad()

# Backward pass
loss.backward()

# Update the weights
optimizer = optim.SGD(dcn.parameters(), lr=0.1)
optimizer.step()

# Check if the parameters are the same
for param1, param2 in zip(tmp, dcn.parameters()):
    if not torch.equal(param1, param2):
        print("Parameters are not the same!")
        print("param1:", param1)
        print("param2:", param2)
