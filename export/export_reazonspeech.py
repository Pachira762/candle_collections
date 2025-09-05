import os
import onnx
import onnx.numpy_helper
import huggingface_hub as hf
import safetensors.numpy


def load_model(repo_id: str, filename: str) -> onnx.ModelProto:
    basedir = hf.snapshot_download(repo_id)
    path = os.path.join(basedir, filename)
    return onnx.load_model(path)


def get_encoder_weight():
    model = load_model(
        "reazon-research/reazonspeech-k2-v2", "encoder-epoch-99-avg-1.onnx"
    )

    rename_map = {}
    transpose_target = []

    for i, node in enumerate(model.graph.node):
        if node.op_type == "MatMul":
            next = model.graph.node[i + 1]

            if next.op_type == "Add":  # Linear with bias
                weight_name = node.input[1]
                # e.g. encoder_embed.out.bias
                bias_name = next.input[0]

                if not weight_name.startswith("onnx::") or ".bias" not in bias_name:
                    print(f"Invalid MatMul pattern\n{node}\n{next}")
                    exit(-1)

                new_name = bias_name.replace(".bias", ".weight")
                rename_map[weight_name] = new_name
                transpose_target.append(new_name)

            else:  # Lienar without bias
                weight_name = node.input[1]

                # e.g. /encoder/0/layers.0/self_attn_weights/linear_pos/MatMul
                # e.g. /encoder/1/encoder/0/self_attn_weights/linear_pos/MatMul
                parts = node.name.split("/")[1:-1]
                parts[0] = "encoder.encoders"
                if parts[2] == "encoder":
                    parts[2] = "encoder.layers"
                new_name = ".".join(parts) + ".weight"
                rename_map[weight_name] = new_name
                transpose_target.append(new_name)

        elif node.op_type == "Mul" and node.input[1].startswith("onnx::"):
            if "/downsample/" in node.name:
                weight_name = node.input[1]

                # e.g. /encoder/1/downsample/Mul_2
                index = node.name.split("/")[2]
                new_name = f"encoder.encoders.{index}.downsample.weight"
                rename_map[weight_name] = new_name

            else:
                # e.g. /encoder/downsample_output/Mul_2
                weight_name = node.input[1]
                new_name = "encoder.downsample_output.weight"
                rename_map[weight_name] = new_name

    tensors = {}
    for init in model.graph.initializer:
        name = init.name
        tensor = onnx.numpy_helper.to_array(init)

        if name in rename_map:
            name = rename_map[name]

        if name in transpose_target:
            tensor = tensor.T

        if tensor.shape == ():
            tensor = tensor.reshape(1)

        tensors[f"encoder.{name}"] = tensor

    return tensors


def get_decoder_weight():
    model = load_model(
        "reazon-research/reazonspeech-k2-v2", "decoder-epoch-99-avg-1.onnx"
    )

    tensors = {}
    for init in model.graph.initializer:
        name = init.name
        tensor = onnx.numpy_helper.to_array(init)

        if tensor.shape == ():
            tensor = tensor.reshape(1)

        if not name.startswith("decoder."):
            name = f"decoder.{name}"

        tensors[name] = tensor

    return tensors


def get_joiner_weight():
    model = load_model(
        "reazon-research/reazonspeech-k2-v2", "joiner-epoch-99-avg-1.onnx"
    )

    tensors = {}
    for init in model.graph.initializer:
        name = init.name
        tensor = onnx.numpy_helper.to_array(init)

        if tensor.shape == ():
            tensor = tensor.reshape(1)

        if not name.startswith("joiner."):
            name = f"joiner.{name}"

        tensors[name] = tensor

    return tensors


def export():
    tensors = {}
    tensors.update(get_encoder_weight())
    tensors.update(get_decoder_weight())
    tensors.update(get_joiner_weight())

    filename = os.path.join("models", "reazonspeech-k2-v2.safetensors")
    safetensors.numpy.save_file(tensors, filename)


if __name__ == "__main__":
    export()
