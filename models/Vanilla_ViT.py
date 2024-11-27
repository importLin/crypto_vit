import timm
from torch import nn


class VanillaViT_timm(nn.Module):
    def __init__(self, backbone_name, n_classes, frozen_backbone=True):
        super(VanillaViT_timm, self).__init__()
        self.model = timm.create_model(backbone_name, pretrained=True)
        self.n_classes = n_classes
        if frozen_backbone:
            for name, param in self.model.named_parameters():
                    param.requires_grad = False

        self.model.head = nn.Linear(self.model.head.in_features, out_features=n_classes)

    def forward(self, x):
        return self.model(x)


    def save_model_parameters(self):
        return self.model.state_dict()

    def load_model_parameters(self, state_dict):
        self.model.load_state_dict(state_dict)



def main():
    model = VanillaViT_timm(10)
    # img = torch.rand(1, 3, 224, 224)
    # logits = model(img)
    # print(logits.log_softmax(dim=-1))


if __name__ == '__main__':
    main()



