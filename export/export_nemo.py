import safetensors.torch
import nemo.collections.asr as nemo_asr
import json
from omegaconf import OmegaConf
import os.path


def dump(model_name: str):
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    model.eval()
    model_name = model_name.split("/")[1]

    with open(f"{model_name}.json", "w", encoding="utf-8") as f:
        config = OmegaConf.to_container(model.cfg, resolve=True)
        json.dump(config, f, ensure_ascii=False, indent=4)

    with open(f"{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"{model}")


def export(model_name: str):
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    model.eval()
    model_name = model_name.split("/")[1]

    tensors = model.state_dict()
    filename = os.path.join("models", f"{model_name}.safetensors")
    config = OmegaConf.to_container(model.cfg, resolve=True)
    metadata = {"config": json.dumps(config, ensure_ascii=False, indent=4)}
    safetensors.torch.save_file(tensors, filename, metadata)

    print(f"save {model_name}")


def main():
    dump("nvidia/parakeet-tdt-0.6b-v2")
    export("nvidia/parakeet-tdt-0.6b-v2")


if __name__ == "__main__":
    main()
