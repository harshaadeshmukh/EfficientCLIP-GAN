import os
import sys
import io
import torch
import torchvision
import clip
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from utils import load_model_weights
from model import NetG, CLIP_TXT_ENCODER

# checking the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# repositiory of the model
repo_id = "VinayHajare/EfficientCLIP-GAN"
file_name = "saved_models/state_epoch_1480.pth"

# clip model wrapped with the custom encoder
clip_text = "ViT-B/32"
clip_model, preprocessor = clip.load(clip_text, device=device)
clip_model = clip_model.eval()
text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)

# loading the model from the repository and extracting the generator model
model_path = hf_hub_download(repo_id = repo_id, filename = file_name)
checkpoint = torch.load(model_path, map_location=torch.device(device))
netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
generator = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)

# Function to generate images from text
def generate_image_from_text(caption, batch_size=4):
    # Create the noise vector
    noise = torch.randn((batch_size, 100)).to(device)
    with torch.no_grad():
        # Tokenize caption
        tokenized_text = clip.tokenize([caption]).to(device)
        # Extract the sentence and word embedding from Custom CLIP ENCODER
        sent_emb, word_emb = text_encoder(tokenized_text)
        # Repeat the sentence embedding to match the batch size
        sent_emb = sent_emb.repeat(batch_size, 1)
        # generate the images
        generated_images = generator(noise, sent_emb, eval=True).float()

        # Convert the tensor images to PIL format
        pil_images = []
        for image_tensor in generated_images.unbind(0):
            # Rescale tensor values to [0, 1]
            image_tensor = image_tensor.data.clamp(-1, 1)
            image_tensor = (image_tensor + 1.0) / 2.0

            # Convert tensor to numpy array
            image_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()

            # Clip numpy array values to [0, 1]
            image_numpy = np.clip(image_numpy, 0, 1)

            # Create a PIL image from the numpy array
            pil_image = Image.fromarray((image_numpy * 255).astype(np.uint8))

            pil_images.append(pil_image)

        return pil_images

# Function to generate images from text
def generate_image_from_text_with_persistent_storage(caption, batch_size=4):
    # Create the noise vector
    noise = torch.randn((batch_size, 100)).to(device)
    with torch.no_grad():
        # Tokenize caption
        tokenized_text = clip.tokenize([caption]).to(device)
        # Extract the sentence and word embedding from Custom CLIP ENCODER
        sent_emb, word_emb = text_encoder(tokenized_text)
        # Repeat the sentence embedding to match the batch size
        sent_emb = sent_emb.repeat(batch_size, 1)
        # generate the images
        generated_images = generator(noise, sent_emb, eval=True).float()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            pil_images = []
            for idx, image_tensor in enumerate(generated_images.unbind(0)):
                # Save the image tensor to a temporary file
                image_path = os.path.join(temp_dir, f"image_{idx}.png")
                torchvision.utils.save_image(image_tensor.data, image_path, value_range=(-1, 1), normalize=True)

                # Load the saved image using Pillow
                pil_image = Image.open(image_path)
                pil_images.append(pil_image)

            return pil_images
