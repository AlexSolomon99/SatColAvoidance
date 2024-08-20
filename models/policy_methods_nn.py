import torch


class SatColAvoidPolicy(torch.nn.Module):

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

        # the output has 6 components, defining 3 normal distributions. The first 3 elements of the output are the mean
        # and the next 3 are the corresponding stds of the 3 distributions.
        out[:3] = torch.tanh(out[:3])
        out[3:] = self.action_std_activation_function(out[3:])
        return out


    @staticmethod
    def action_std_activation_function(x):
        return torch.relu(x) + 1e-5
