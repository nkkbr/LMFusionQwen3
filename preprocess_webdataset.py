# conda activate lmfusion
# cd /disk/extra_c26e245c/fengqi/LMFusion
# python preprocess_webdataset.py --data_file "/disk/extra_c26e245c/fengqi/U_06/pixparse/cc12m-wds/cc12m-train-0000.tar" --output_dir "/disk/extra_c26e245c/fengqi/U_06/output/cc12m/0000" --num_proc 8 --batch_size 16

import torch
import logging
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoTokenizer
from typing import Dict, List
import webdataset as wds
from PIL import Image 
import io
from typing import Union, Optional
from diffusers.models.modeling_outputs import AutoencoderKLOutput
import argparse
from datasets import load_dataset, Dataset, Features, Value
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# From https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L86-L96
def retrieve_latents(
    encoder_output: Union[torch.Tensor, AutoencoderKLOutput], 
    generator: Optional[torch.Generator] = None, 
    sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    
def center_crop_square_and_resize(
    image: Image.Image, 
    target_size=(256, 256)
) -> Image.Image:
    width, height = image.size
    if (width, height) == target_size:
        return image
    
    min_edge = min(width, height)
    
    left = (width - min_edge) // 2
    top = (height - min_edge) // 2
    right = left + min_edge
    bottom = top + min_edge
    
    image = image.crop((left, top, right, bottom))
    
    image = image.resize(target_size, Image.LANCZOS)  # Image.BICUBIC can also be used
    return image

def get_map_function(
    device:torch.device='cuda',
    model_name_or_path: str = "Qwen/Qwen3-8B", 
    vae_name_or_path: str = "stabilityai/sd-vae-ft-mse", 
):

    def process_batch(
        batch: Dict[str, List]
    ) -> Dict[str, List]:
        
        global process_models
        if 'process_models' not in globals():
            logger.info(f"Initializing models for process: {os.getpid()}")
            process_models = {
                "tokenizer": AutoTokenizer.from_pretrained(model_name_or_path),
                "vae": AutoencoderKL.from_pretrained(vae_name_or_path).to(device),
                "image_processor": VaeImageProcessor(vae_scale_factor=8)
            }
        
        tokenizer = process_models["tokenizer"]
        vae = process_models["vae"]
        image_processor = process_models['image_processor']
        

        image_bytes_list = batch['jpg']
        
        images = []
        valid_texts = []
        original_texts = batch['txt']

        for i, image_bytes in enumerate(image_bytes_list):
            if image_bytes is None:
                logger.warning("Skipping a null image entry.")
                continue
            try:
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
                valid_texts.append(original_texts[i])
            except Exception as e:
                logger.warning(f"Skipping a corrupt or unreadable image. Error: {e}")
                continue

        if not images:
             return {"input_ids_list": [], "clean_latents": []}


        images = [image.convert("RGB") if image.mode != "RGB" else image for image in images]
        images = [center_crop_square_and_resize(image) for image in images]

        pixel_values = image_processor.preprocess(images, height=256, width=256).to(device)
        
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        latents_cpu_fp32 = latents.cpu().to(torch.float32).numpy()

        texts = batch['txt']
        # texts = [text.decode('utf-8') for text in texts]
        texts = [tokenizer.encode(
            text,
            # return_tensors='pt' # It's said that int is more efficient than torch.tensor
            ) for text in texts]

        if len(texts) != len(latents_cpu_fp32):
             logger.error(f"Mismatch between number of texts ({len(texts)}) and latents ({len(latents_cpu_fp32)}). Skipping batch.")
             return {"input_ids_list": [], "clean_latents": []}

        return {
            "input_ids_list":texts,
            "clean_latents":latents_cpu_fp32
        }
    
    return process_batch


def main():

    parser = argparse.ArgumentParser(description="Preprocess image-text pair data for LMFusion training.")
    parser.add_argument("--data_file", type=str, default="/disk/extra_c26e245c/fengqi/U_06/pixparse/cc12m-wds/cc12m-train-0000.tar")
    parser.add_argument("--output_dir", type=str, default="/disk/extra_c26e245c/fengqi/U_06/output/cc12m/0000")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use for preprocessing.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for VAE encoding.")

    args = parser.parse_args()

    logger.info("Starting dataset preprocessing...")
    logger.info(f"Loading raw WebDataset from: {args.data_file}")

    forced_features = Features({
        'jpg': Value('binary'),
        'txt': Value('string'),
        '__key__': Value('string'), 
    })

    raw_dataset = load_dataset(
        "webdataset",
        data_files={"train": args.data_file},
        features=forced_features,  
        streaming=False,
        split="train" 
    )

    map_function = get_map_function()
    logger.info(f"Applying map function with batch size {args.batch_size} and {args.num_proc} processes...")
    
    processed_dataset = raw_dataset.map(
        map_function,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=raw_dataset.column_names
    )
    
    processed_dataset.save_to_disk(args.output_dir)
    logger.info(f"Successfully saved processed dataset to {args.output_dir}")


if __name__ == "__main__":
    main()