import cupy as cp

# Check the number of available GPUs
device_count = cp.cuda.runtime.getDeviceCount()
print(f"Number of GPUs: {device_count}")

if device_count > 0:
    # Access the first GPU and print its name
    with cp.cuda.Device(0):  # Specify device 0 (first GPU)
        print("CUDA is available!")
        # Create two large arrays on the GPU
        a = cp.random.rand(1000000)  # Create random array on GPU
        b = cp.random.rand(1000000)  # Create another random array on GPU

        # Perform element-wise addition on the GPU
        c = a + b

        # Print the first few elements of the result
        print("First 5 elements of the result:", c[:5])
else:
    print("No CUDA devices detected.")