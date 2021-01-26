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
from tumbocr.data.data_utils import get_vocabulary
from tumbocr.models.create_model import create_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_torch_model(congfig_path):
    cfg = EasyDict(load_config(congfig_path))
    cfg.Global.device = "cuda"
    char2id,id2char = get_vocabulary(cfg.Global.dict_path)
    cfg.Global.num_classes = len(char2id)
    model = create_model(cfg.Model.arch)(cfg).cuda()
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
def check_model(config,pretrain_path=None):
    torch_model = load_torch_model(config)
    # print(torch_model)
    input_tensor = torch.randn(2,3,48,160).cuda()
    target_tensor = torch.zeros(2,97).cuda()
    output = torch_model(input_tensor,target_tensor)
    print(output.shape)
    # torch.save(torch_model.state_dict(), "lite_model/model.pth")

if __name__ == '__main__':
    congfig_path = sys.argv[1]
    pretrain_path = sys.argv[2]
    check_model(congfig_path,pretrain_path)
