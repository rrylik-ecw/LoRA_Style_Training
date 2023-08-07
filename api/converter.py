import torch
from safetensors.torch import load_file, save_file
from diffusers import StableDiffusionPipeline


class LoRAConverter:
    def __init__(self, model_id):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.LORA_PREFIX_UNET = 'lora_unet'

    def convert_name_to_bin(self, name):
        # Implementation of convert_name_to_bin function
        new_name = name.replace(self.LORA_PREFIX_UNET + '_', '')
        new_name = new_name.replace('.weight', '')

        parts = new_name.split('.')
        sub_parts = parts[0].split('_')

        new_sub_parts = ""
        for i in range(len(sub_parts)):
            if sub_parts[i] in ['block', 'blocks', 'attentions'] or sub_parts[i].isnumeric() or 'attn' in sub_parts[i]:
                if 'attn' in sub_parts[i]:
                    new_sub_parts += sub_parts[i] + ".processor."
                else:
                    new_sub_parts += sub_parts[i] + "."
            else:
                new_sub_parts += sub_parts[i] + "_"

        new_sub_parts += parts[1]
        new_name = new_sub_parts + '.weight'

        return new_name

    def safetensors_to_bin(self, safetensor_path, bin_path):
        bin_state_dict = {}
        safetensors_state_dict = load_file(safetensor_path)

        for key_safetensors in safetensors_state_dict:
            if 'text' in key_safetensors:
                continue
            if 'unet' not in key_safetensors:
                continue
            if 'transformer_blocks' not in key_safetensors:
                continue
            if 'ff_net' in key_safetensors or 'alpha' in key_safetensors:
                continue
            key_bin = self.convert_name_to_bin(key_safetensors)
            bin_state_dict[key_bin] = safetensors_state_dict[key_safetensors]

        torch.save(bin_state_dict, bin_path)

    def convert_name_to_safetensors(self, name):
        # Implementation of convert_name_to_safetensors function
        parts = name.split('.')

        for i in range(len(parts)):
            if parts[i].isdigit():
                parts[i] = '_' + parts[i]
            if "to" in parts[i] and "lora" in parts[i]:
                parts[i] = parts[i].replace('_lora', '.lora')

        new_parts = []
        for i in range(len(parts)):
            if i == 0:
                new_parts.append(self.LORA_PREFIX_UNET + '_' + parts[i])
            elif i == len(parts) - 2:
                new_parts.append(parts[i] + '_to_' + parts[i + 1])
                new_parts[-1] = new_parts[-1].replace('_to_weight', '')
            elif i == len(parts) - 1:
                new_parts[-1] += '.' + parts[i]
            elif parts[i] != 'processor':
                new_parts.append(parts[i])
        new_name = '_'.join(new_parts)
        new_name = new_name.replace('__', '_')
        new_name = new_name.replace('_to_out.', '_to_out_0.')
        return new_name

    def bin_to_safetensors(self, bin_path, safetensor_path):
        bin_state_dict = torch.load(bin_path)
        safetensors_state_dict = {}

        for key_bin in bin_state_dict:
            key_safetensors = self.convert_name_to_safetensors(key_bin)
            safetensors_state_dict[key_safetensors] = bin_state_dict[key_bin]

        save_file(safetensors_state_dict, safetensor_path)

    def convert_to_bin(self, safetensor_path, bin_path):
        self.safetensors_to_bin(safetensor_path, bin_path)

    def convert_to_safetensors(self, bin_path, safetensor_path):
        self.bin_to_safetensors(bin_path, safetensor_path)