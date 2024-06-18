# GAI_Project4

# DIP with DDPM-Inspired Supervision

This project explores an innovative approach to enhance Deep Image Prior (DIP) training using DDPM-inspired gradual denoising steps. The goal is to determine the optimal early stopping point and improve the reconstruction quality by leveraging hierarchical denoising.

## Requirements

Ensure you have the following libraries installed:

- Python 3.x
- NumPy
- PyTorch
- torchvision
- matplotlib
- scikit-image

You can install the required libraries using the following command:

```bash
pip install numpy torch torchvision matplotlib scikit-image
```

## File Description

- `main.ipynb`: Jupyter Notebook containing the implementation of the DIP model with DDPM-inspired supervision. This notebook includes the following sections:
  - Loading the target image
  - Adding noise at different levels
  - Training the DIP model
  - Monitoring reconstruction quality
  - Ablation studies to determine optimal parameters

## Usage Instructions

### Step 1: Load the Notebook

1. Open Jupyter Notebook and navigate to the directory containing `main.ipynb`.
2. Launch the notebook.

### Step 2: Load the Target Image

1. The notebook begins with loading a target image. Ensure you have an image ready for processing.
2. Modify the path to your target image in the cell where the image is loaded.

```python
from skimage import io

# Load your target image
target_image = io.imread('path_to_your_image.jpg')
```

### Step 3: Adding Noise at Different Levels

1. The notebook contains code to add noise to the target image at various levels.
2. You can modify the noise levels by adjusting the parameters in the corresponding cell.

```python
def add_noise(image, noise_level):
    noise = torch.randn_like(image) * noise_level
    return image + noise

# Define noise levels
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
noisy_images = [add_noise(target_image, nl) for nl in noise_levels]
```

### Step 4: Train the DIP Model

1. Follow the cells to set up and train the DIP model using the noisy images.
2. Adjust training parameters such as learning rate and number of epochs as needed.

```python
# Define the DIP model architecture and training parameters
dip_model = DIPModel()
optimizer = optim.Adam(dip_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for noisy_image in noisy_images:
        optimizer.zero_grad()
        output = dip_model(noisy_image)
        loss = criterion(output, noisy_image)
        loss.backward()
        optimizer.step()
```

### Step 5: Monitor Reconstruction Quality

1. The notebook includes cells to monitor the reconstruction quality using PSNR and SSIM metrics.
2. Use these metrics to analyze the performance of the DIP model at each denoising stage.

```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Monitor reconstruction quality
psnr_values = [psnr(output.cpu().detach().numpy(), noisy_image.cpu().detach().numpy()) for output, noisy_image in zip(outputs, noisy_images)]
ssim_values = [ssim(output.cpu().detach().numpy(), noisy_image.cpu().detach().numpy(), multichannel=True) for output, noisy_image in zip(outputs, noisy_images)]
```

### Step 6: Conduct Ablation Studies

1. The notebook includes sections for conducting ablation studies to determine the impact of different noise levels, denoising schedules, and architectures.
2. Modify the parameters and observe the results to identify the optimal configurations.

```python
# Example of varying noise levels
for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
    noisy_images = [add_noise(target_image, noise_level)]
    # Train and evaluate model
    # ...

# Example of varying architectures
for architecture in [SimpleEncoderDecoder(), UNet(), ResNet()]:
    dip_model = architecture
    # Train and evaluate model
    # ...
```

### Conclusion

By following these steps, you can effectively use the provided notebook to implement and evaluate the DIP model with DDPM-inspired supervision. The ablation studies will help you understand the impact of various parameters and guide you in choosing the best configurations for your specific use case.
