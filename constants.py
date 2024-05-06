# Highway
HW_IMAGE_HEIGHT = 64
HW_IMAGE_WIDTH = 128
VEHICLE_COUNT = 10
ENV_FEATURES_SIZE = 7
OCCUPANCY_INPUT_SIZE = 847

# Cart Pole
SCALE_FACTOR = 8
CP_IMAGE_HEIGHT = 400 // SCALE_FACTOR  # 400
CP_IMAGE_WIDTH = 600 // SCALE_FACTOR  # 600
CP_ACTION_SPACE = 2
CP_OBS_SPACE = 4
CP_INPUT_SIZE = CP_ACTION_SPACE * CP_OBS_SPACE

# Car Racing
CR_IMAGE_HEIGHT = 96
CR_IMAGE_WIDTH = 96



# import torch
# print(torch.version.cuda)
# print(torch.cuda.is_available())  # Should print True if CUDA is properly installed and a compatible GPU is found
#
# print(torch.cuda.get_device_name(0))

# import site
# print(site.getsitepackages())
