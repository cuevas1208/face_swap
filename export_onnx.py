# export_onnx.py
import torch
from models_anonymizer import GeneratorUNet

def export_decode_to_onnx(ckpt_path, onnx_out, batch_size=1, bottleneck_dim=512, opset=13):
    device = torch.device('cpu')
    G = GeneratorUNet(bottleneck_dim=bottleneck_dim).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if 'G' in sd:
        sdG = sd['G']
    else:
        sdG = sd
    G.load_state_dict(sdG)
    G.eval()
    # create dummy latent
    z = torch.randn(batch_size, G.fc.in_features, device=device)  # careful: fc maps base*8 -> bottleneck_dim; we used fc.in_features below
    # but that structure: fc: base*8 -> bottleneck_dim, so decode takes z shape (B, bottleneck_dim)
    z = torch.randn(batch_size, G.fc.out_features, device=device)
    # Actually find bottleneck size:
    bdim = G.fc.out_features
    z = torch.randn(batch_size, bdim)
    # call decode_from_latent
    torch.onnx.export(G.decode_from_latent, (z,), onnx_out, opset_version=opset,
                      input_names=['z'], output_names=['out'],
                      dynamic_axes={'z': {0: 'batch'}, 'out': {0: 'batch', 2: 'H', 3: 'W'}})
    print("Exported ONNX to", onnx_out)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--onnx-out', required=True)
    p.add_argument('--bottleneck-dim', type=int, default=512)
    args = p.parse_args()
    export_decode_to_onnx(args.ckpt, args.onnx_out, batch_size=1, bottleneck_dim=args.bottleneck_dim)
