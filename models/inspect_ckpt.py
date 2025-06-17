import torch

def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, dict):
                print(f"[{k}] has {len(v)} keys")
            elif isinstance(v, torch.Tensor):
                print(f"[{k}] is tensor with shape {v.shape}")
            else:
                print(f"[{k}] is {type(v)}")
    elif isinstance(ckpt, torch.nn.Module):
        print("Loaded a model object directly")
    else:
        print("Unknown structure:", type(ckpt))

inspect_checkpoint("/home/ivpl-d29/sichoi/Emo/unidiffuser/models/uvit_v1.pth")
inspect_checkpoint("/home/ivpl-d29/sichoi/Emo/unidiffuser/si_finetuning_2/leg_checkpoint_step_4000.pth")
