import torch

if torch.cuda.is_available():
	device_count = torch.cuda.device_count()
	print(f"Number of CUDA devices available: {device_count}")
	for i in range(device_count):
		print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
	print("CUDA is not available. Check your CUDA installation.")