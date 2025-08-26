# Our pre-processed tensors in latent space, obtained after processing images with a VAE.
# We also pre-process the input_ids for each caption, without any templates.
# The Dataset reads these two from the file according to the index.
# The DataCollator then packages this information for a batch.

from typing import Sequence, Dict, Literal
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_from_disk

class LMFusionQwen3Dataset(Dataset):
    """
    Input is the already processed clean_latent and text.
    Output is clean_latent and input_ids.
    """
    def __init__(
        self,
        data_path: str,
        training_phase: Literal['pretrain', 'finetune'],
    ):
        super().__init__()
        self.data_path=data_path
        self.training_phase=training_phase
        self.dataset = load_from_disk(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        if self.training_phase == 'pretrain':

            # Get from the file
            clean_latent = torch.tensor(self.dataset[index]['clean_latents']).unsqueeze(0) # (1, 4, 32, 32)
            input_ids = torch.tensor(self.dataset[index]['input_ids_list']).unsqueeze(0) # (batch_size=1,seqLen)

            data_dict = dict(
                input_ids = input_ids,
                clean_latent = clean_latent
            )
            return data_dict
        elif self.training_phase == 'finetune':
            pass
        else:
            raise ValueError(f"Invalid task_name: '{self.training_phase}'. Expected 'pretrain' or 'finetune'.")


@dataclass
class DataCollatorForLMFusionQwen3Dataset(object):

    def __call__(
        self,
        instances: Sequence[Dict]
    ) -> Dict[str, torch.Tensor]:
                
        input_ids_list = [item['input_ids'] for item in instances]
        clean_latent_list = [item['clean_latent'] for item in instances]
        clean_latents = torch.cat(clean_latent_list, dim=0)
        
        return dict(
            input_ids_list=input_ids_list,
            clean_latents=clean_latents
        )