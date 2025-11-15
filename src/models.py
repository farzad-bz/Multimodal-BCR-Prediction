import torch
import torch.nn as nn
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from collections import OrderedDict
from huggingface_hub import hf_hub_download

class SurvivalModelMM(nn.Module):

    """
    Modular multimodal survival model.

    - Each modality has its own Linear -> LayerNorm -> ReLU -> Dropout block
    that maps from its raw input dim to a shared embedding dim d_emb.
    - Any subset of modalities can be provided at forward().
    - Embeddings of all provided modalities are averaged, then passed to
    a small MLP to predict risk.

    Example modalities:
        modalities = {
            "clinical": 6,     # len(features)
            "t2": 768,         # M3D-CLIP T2 emb
            "hbv": 768,        # M3D-CLIP HBV emb
            "adc": 768,        # M3D-CLIP ADC emb
        }
    """
    def __init__(
        self,
        modalities: dict,   # e.g. {"clinical": 6, "t2": 768, "hbv": 768, "adc": 768}
        d_emb: int = 16,          # embedding dim for EACH modality
        dropout: float = 0.2,
    ):
        super().__init__()

        self.modalities = modalities
        self.d_emb = d_emb

        # One projection + norm per modality
        self.proj = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for name, in_dim in modalities.items():
            self.proj[name] = nn.Linear(in_dim, d_emb)
            self.norm[name] = nn.LayerNorm(d_emb)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Fusion head operates on a single fused embedding of size d_emb
        self.fusion = nn.Sequential(
            nn.Linear(d_emb, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, inputs: dict):
        """
        inputs: dict mapping modality_name -> tensor
            - Each tensor should be of shape (B, modalities[name]).
            - All provided modalities must have the same batch size B.

        Examples:
            risk = model({"clinical": x_clin})
            risk = model({"clinical": x_clin, "t2": t2_emb})
            risk = model({"clinical": x_clin, "t2": t2_emb, "hbv": hbv_emb})
            risk = model({"t2": t2_emb, "adc": adc_emb})
        """
        assert len(inputs) > 0, "At least one modality must be provided"

        embs = []
        for name, x in inputs.items():
            if name not in self.proj:
                raise ValueError(f"Unknown modality '{name}'. "
                                f"Known modalities: {list(self.proj.keys())}")
            z = self.proj[name](x)          # (B, d_emb)
            z = self.norm[name](z)          # (B, d_emb)
            z = self.act(z)
            z = self.dropout(z)
            embs.append(z)

        # fuse: mean over available modalities -> (B, d_emb)
        if len(embs) == 1:
            fused = embs[0]
        else:
            fused = torch.stack(embs, dim=0).mean(dim=0)

        risk = self.fusion(fused).squeeze(-1)  # (B,)
        return risk
    
    


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.conv_seg(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model



def get_image_encoder(cfg, device):
    
    if cfg.image_encoder.type == "M3D-CLIP":
        M3D_model = AutoModel.from_pretrained(
            cfg.image_encoder.pretrained_path,
            trust_remote_code=True
        )
        M3D_model = M3D_model.to(device=device)
        M3D_model.requires_grad_(False)
        M3D_model.eval()
        return M3D_model
    
    elif cfg.image_encoder.type == "MedicalNet":
        model = resnet10(
                        shortcut_type='B',
                        no_cuda=False)
        model.cuda()

        model_path = hf_hub_download(repo_id=cfg.image_encoder.pretrained_path, filename="resnet_10_23dataset.pth")

        ckpt = torch.load(model_path, map_location="cpu")['state_dict']

        old_state_dict = ckpt
        new_state_dict = OrderedDict()

        for k, v in old_state_dict.items():
            # remove "module." if it exists
            if k.startswith("module."):
                new_k = k[len("module."):]
            else:
                new_k = k
            new_state_dict[new_k] = v


        model.load_state_dict(new_state_dict)
        model = model.to(device=device)
        model.requires_grad_(False)
        model.eval()
        return model
    
    else:
        raise Exception("image_encoder.type should either be M3D-CLIP or MedicaNet")