from models.resnet_model import *


inps, outs = [],[]
def layer_hook(module, inp, out):
    inps.append(inp[0].data.cpu().numpy())
    outs.append(out.data.cpu().numpy())
    print(outs[0].shape)

class ResNet18(ResNestBaseModel):
    def __init__(self, num_classes, pre_trained=False):
        super(ResNet18, self).__init__(num_classes, pre_trained)
        self.resnet_18 = models.resnet18(pretrained=self.pre_trained)
        self.resnet_18.fc = nn.Linear(self.resnet_18.fc.in_features, self.num_classes)

    def forward(self, x):

        # hook = self.resnet_18.layer4.register_forward_hook(layer_hook)
        y = self.resnet_18(x)
        # hook.remove()
        return y
