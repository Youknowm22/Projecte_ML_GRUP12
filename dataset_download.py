import kagglehub
import os

# Download latest version

target = os.path.join(os.getcwd(), "Dataset")

path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions", output_dir=target)

print("Dataset downloaded", path)