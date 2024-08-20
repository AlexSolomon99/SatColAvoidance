import torch


class QNetwork(torch.nn.Module):
    def __init__(self, conf: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn_net = torch.nn.Sequential(
            torch.nn.LayerNorm(conf["init_layer"]),
            torch.nn.Linear(conf["init_layer"], conf["hidden_layer_1"]),
            torch.nn.ReLU(),
            torch.nn.Linear(conf["hidden_layer_1"], conf["hidden_layer_2"]),
            torch.nn.ReLU(),
            torch.nn.Linear(conf["hidden_layer_2"], conf["output_layer"])
        )

    def forward(self, x):
        out = self.nn_net(x)
        out = torch.tanh(out)

        return out
