import torch


class SatColAvoidPolicy(torch.nn.Module):

    def __init__(self, conf: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn_net = torch.nn.Sequential(
            torch.nn.Linear(conf["init_layer"], conf["hidden_layer"]),
            torch.nn.ReLU(),
            torch.nn.Linear(conf["hidden_layer"], conf["output_layer"])
        )

    def forward(self, x):
        out = self.nn_net(x)
        return out
