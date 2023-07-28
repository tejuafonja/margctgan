import numpy as np
import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
    init,
)


class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, inputs):
        return self.seq(inputs)


class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, inputs):
        return self.seq(inputs)


class Classifier(Module):
    def __init__(self, side, layers, transformer, device):
        super(Classifier, self).__init__()
        self.side = side
        self.transformer = transformer
        self.seq = Sequential(*layers)

        index = transformer.output_dimensions - 1
        target_index_in_raw_data = self.transformer._target_index_in_raw_data

        if transformer._target_index is None or target_index_in_raw_data is None:
            self.valid = False
        elif (
            self.transformer._column_transform_info_list[
                target_index_in_raw_data
            ].column_type
            != "discrete"
        ):
            self.valid = False
        elif (
            len(
                self.transformer._column_transform_info_list[
                    target_index_in_raw_data
                ].transform.classes_
            )
            > 2
        ):
            self.valid = False
        else:
            self.valid = True
            index = transformer._target_index

        masking = np.ones((1, 1, side, side), dtype="float32")
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, inputs):
        label = (inputs[:, :, self.r, self.c].view(-1) + 1) / 2
        inputs = inputs * self.masking.expand(inputs.size())
        return self.seq(inputs).view(-1), label


def determine_layer_dims(side, num_channels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    return layer_dims


def classifier_layers(layer_dims):
    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(
                in_channels=prev[0],
                out_channels=curr[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(num_features=curr[0]),
            LeakyReLU(0.2, inplace=True),
        ]

    layers_C += [
        Conv2d(
            in_channels=layer_dims[-1][0],
            out_channels=1,
            kernel_size=layer_dims[-1][1],
            stride=1,
            padding=0,
        )
    ]
    return layers_C


def generator_layers(random_dim, layer_dims):
    layers_G = [
        ConvTranspose2d(
            in_channels=random_dim,
            out_channels=layer_dims[-1][0],
            kernel_size=layer_dims[-1][1],
            stride=1,
            padding=0,
            output_padding=0,
            bias=False,
        )
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(num_features=prev[0]),
            ReLU(inplace=True),
            ConvTranspose2d(
                in_channels=prev[0],
                out_channels=curr[0],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=True,
            ),
        ]

    layers_G += [Tanh()]
    return layers_G


def discriminator_layers(layer_dims):
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(
                in_channels=prev[0],
                out_channels=curr[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(num_features=curr[0]),
            LeakyReLU(0.2, inplace=True),
        ]
    layers_D += [
        Conv2d(
            in_channels=layer_dims[-1][0],
            out_channels=1,
            kernel_size=layer_dims[-1][1],
            stride=1,
            padding=0,
        ),
        Sigmoid(),
    ]

    return layers_D


def determine_layers(side, random_dim, num_channels):
    layer_dims = determine_layer_dims(side, num_channels)

    layers_D = discriminator_layers(layer_dims)

    layers_G = generator_layers(random_dim, layer_dims)

    layers_C = classifier_layers(layer_dims)

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)
