import torch
import argparse 
from collections import OrderedDict
from torch.nn import DataParallel
import os
from modeling_infer.deeplab import *
def generate_onnx(model_path, onnx_path):
        checkpoint = torch.load(model_path)
        model = DeepLab(num_classes=2)
        model = model.cuda().eval().half()
        # model.load_state_dict(checkpoint['state_dict'])
        pretrained_dict = checkpoint['state_dict']
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            if k in state_dict:
                model_dict[k] = v.type(torch.float16)
            else:
                print(f'{k} is not in model_dict')
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
        dummy_input = torch.randn(1, 3, 512, 512, device='cuda').half()

        print('start exporting libtorch model')
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save("weights/model_half.pt")
        print('finish exporting')

        # print('start exporting onnx model')
        # torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version = 11)
        # print('finish exporting')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeeplabV3Plus")
    parser.add_argument('--model', type=str, default="weights/checkpoint_convert.pth.tar", help='trained model path')
    parser.add_argument('--onnx', type=str, default="weights/deeplab.onnx", help='onnx store path')
    args = parser.parse_args() 
    generate_onnx(args.model, args.onnx)
