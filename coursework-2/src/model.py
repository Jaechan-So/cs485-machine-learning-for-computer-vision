from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, skip=False):
        super(ResidualBlock, self).__init__()

        self.skip = skip

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.skip:
            x = self.block(x) + x
        else:
            x = self.block(x)

        return x


class FcBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=False):
        super(FcBlock, self).__init__()

        self.dropout = nn.Dropout() if dropout else None
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.activation(self.fc(x))


class LeNet(nn.Module):
    def __init__(
        self,
        conv_channels,  # Channel values for convolution layers.
        fc_sizes,  # Size values for fully connected layers.
        kernel_size,
        skip=False,
        dropout=False,
        norm=None,
    ):
        super(LeNet, self).__init__()

        self.residual_blocks = nn.ParameterList(
            [
                ResidualBlock(3, skip=skip),
                ResidualBlock(3, skip=skip),
            ]
        )

        conv_channels = [3] + conv_channels

        feature_output_size = 224 + (3 - kernel_size) * (len(conv_channels) - 1) * 2
        fc_sizes = [
            feature_output_size * feature_output_size * conv_channels[-1]
        ] + fc_sizes

        self.feature_blocks = nn.ParameterList()
        for i in range(1, len(conv_channels)):
            in_channels = conv_channels[i - 1]
            out_channels = conv_channels[i]

            feature_block = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                )
            ]

            if norm == "BatchNorm":
                feature_block += [nn.BatchNorm2d(out_channels)]
            elif norm == "InstanceNorm":
                feature_block += [nn.InstanceNorm2d(out_channels)]
            elif norm == "LayerNorm":
                conv_output_size = 224 + (3 - kernel_size) * i * 2
                feature_block += [
                    nn.LayerNorm([out_channels, conv_output_size, conv_output_size])
                ]

            feature_block += [
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=1),
            ]

            feature_block = nn.Sequential(*feature_block)
            self.feature_blocks += feature_block

        self.fc_blocks = nn.ParameterList()
        for i in range(1, len(fc_sizes)):
            self.fc_blocks += [
                FcBlock(
                    in_features=fc_sizes[i - 1],
                    out_features=fc_sizes[i],
                    dropout=dropout,
                )
            ]

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)

        for block in self.feature_blocks:
            x = block(x)

        x = x.flatten(start_dim=1)

        for block in self.fc_blocks:
            x = block(x)

        return x
