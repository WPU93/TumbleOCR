import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from easydict import EasyDict
import torch
import torch.onnx
import onnx
import onnxruntime
from tumbocr.utils.utils import from_pretrained,load_config
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_torch_model(config):
    cfg = EasyDict(load_config(config))
    char2id,id2char = get_vocabulary(cfg.Global.dict_path)
    cfg.Global.num_classes = len(char2id)
    model = create_model(cfg.Model.arch)(cfg)
    return model

def torch2onnx(config,pretrain_path,onnx_path):
    torch_model = load_torch_model(config)
    torch_model.eval()
    input_fp32 = torch.rand([1,3,48,160]).cuda()
    out = torch_model(input_fp32)
    torch.onnx.export(model,input_fp32,onnx_path, export_params=True,
                    opset_version=11,do_constant_folding=True, input_names = ['input'],output_names = ['output'])
    
def quantize_model(config,pretrain_path,quantized_path):
    torch_model = load_torch_model(config)
    torch.save(torch_model.state_dict(), "lite_model/model.pth")

    quantized_model = torch.quantization.quantize_dynamic(
        torch_model, {torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM}, dtype=torch.qint8,inplace=True)
    torch.save(quantized_model, quantized_path)
def 
if __name__ == '__main__':
    congfig = load_config(sys.argv[1])
    pretrain_path = sys.argv[2]
    onnx_path = sys.argv[3]
    torch2onnx(congfig,pretrain_path.onnx_path)
