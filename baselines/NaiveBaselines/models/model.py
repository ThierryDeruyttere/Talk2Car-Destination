import torch
import torch.nn as nn

from .resnet import resnet
from .flownet import FlowNetS


class Model(nn.Module):
    def __init__(
            self,
            dataset="Talk2Car",
            encoder_type="ResNet",
            width=300,
            height=200,
            output_dim=2
    ):
        super(Model, self).__init__()
        assert dataset in {
            "Talk2Car", "Talk2Car_Detector"
        }, "Parameter 'dataset' should be one of {'Talk2Car', 'Talk2Car_Detector'}"

        if dataset == "Talk2Car_Detector":
            input_channels = 15
        else:
            input_channels = 28

        encoder_dim = None
        if encoder_type == "FlowNet":
            encoder_dim = 1024
            self.encoder = FlowNetS(
                input_width=width, input_height=height, input_channels=input_channels
            )
        elif "ResNet" in encoder_type:
            if encoder_type == "ResNet":
                encoder_type = "ResNet-18"
            encoder_dim = 1000
            self.encoder = resnet(encoder_type, in_channels=input_channels)

        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim + 768, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x, command_embedding):
        x = self.encoder(x)
        x = self.regressor(torch.cat([x, command_embedding], dim=-1))
        return x