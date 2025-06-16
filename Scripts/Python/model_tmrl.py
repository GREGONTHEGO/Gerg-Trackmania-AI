import torch
from tmrl.config.config_offline import Config
from tmrl.actor_learner import Learner

# Manually configure the necessary options
config = Config(
    RUN_NAME="SAC_4_imgs_pretrained",  # name of the pretrained run
    DEVICE="cpu",                      # or 'cuda' if you want GPU
    INTERFACE="TM20",                 # 'TM20' is default for Trackmania
)

# Create learner using manual config
learner = Learner(config=config, device="cpu")

# Get the actor model
model = learner.actor_policy.model

# Print the raw architecture
print(model)

# Optional: nice summary with layer shapes and param counts
try:
    from torchsummary import summary
    summary(model, input_size=(4, 64, 64))  # (channels=4, H=64, W=64)
except ImportError:
    print("Install torchsummary via: pip install torchsummary")
