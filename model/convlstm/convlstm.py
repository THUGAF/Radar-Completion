import torch
import torch.nn as nn
from .layers import ConvLSTMCell, Down, Up


class Encoder(nn.Module):
    r"""N-layer ConvLSTM Encoder for 5D (S, B, C, H, W) input. Each layer of the encoder includes a ConvLSTM layer 
        and a convolutional layer.
        
    Args:
        in_channels (int): Input channels.
        hidden_channels (list[int]): Channels of convolutional layers.
    """

    def __init__(self, in_channels: int, hidden_channels: list):
        super(Encoder, self).__init__()
        self.num_layers = len(hidden_channels)

        self.down0 = Down(in_channels, hidden_channels[0])
        self.rnn0 = ConvLSTMCell(hidden_channels[0], hidden_channels[0])

        for i in range(1, self.num_layers):
            setattr(self, 'down' + str(i), Down(hidden_channels[i - 1], hidden_channels[i]))
            setattr(self, 'rnn' + str(i), ConvLSTMCell(hidden_channels[i], hidden_channels[i]))

    def forward(self, x):
        # x: 5D tensor (S, B, C, H, W)
        h_list = [None] * self.num_layers
        c_list = [None] * self.num_layers

        for i in range(x.size(0)):
            y = self.down0(x[i])
            h_list[0], c_list[0] = self.rnn0(y, h_list[0], c_list[0])
            for j in range(1, self.num_layers):
                y = getattr(self, 'down' + str(j))(h_list[j - 1])
                h_list[j], c_list[j] = getattr(self, 'rnn' + str(j))(y, h_list[j], c_list[j])
        
        return h_list, c_list


class Forecaster(nn.Module):
    r"""N-layer ConvLSTM forecaster for 5D (S, B, C, H, W) input. Each layer of the forecaster includes a ConvLSTM layer 
        and a bilinear-resize-convolutional layer.

    Args:
        forecast_steps (int): Forecast steps.
        out_channels (int): Output channels.
        hidden_channels (list[int]): Channels of convolutional layers.
    """

    def __init__(self, forecast_steps: int, out_channels: int, hidden_channels: list):
        super(Forecaster, self).__init__()
        self.num_layers = len(hidden_channels)
        self.forecast_steps = forecast_steps

        for i in range(1, self.num_layers):
            setattr(self, 'rnn' + str(self.num_layers - i), ConvLSTMCell(hidden_channels[-i], hidden_channels[-i]))
            setattr(self, 'up' + str(self.num_layers - i), Up(hidden_channels[-i], hidden_channels[-i-1]))
        
        self.rnn0 = ConvLSTMCell(hidden_channels[0], hidden_channels[0])
        self.up0 = Up(hidden_channels[0], out_channels)
    
    def forward(self, h_list, c_list):
        output = []
        
        x = torch.zeros_like(h_list[-1], device=h_list[-1].device)
        for i in range(self.forecast_steps):
            h_list[-1], c_list[-1] = getattr(self, 'rnn' + str(self.num_layers - 1))(x, h_list[-1], c_list[-1])
            y = getattr(self, 'up' + str(self.num_layers - 1))(h_list[-1])
            for j in range(1, self.num_layers):
                h_list[-j-1], c_list[-j-1] = getattr(self, 'rnn' + str(self.num_layers - j - 1))(y, h_list[-j-1], h_list[-j-1])
                y = getattr(self, 'up' + str(self.num_layers - j - 1))(h_list[-j-1])
            output.append(y)
    
        # output: 5D tensor (S_out, B, C, H, W)
        output = torch.stack(output, dim=0)
        output = torch.relu(output)
        return output


class EncoderForecaster(nn.Module):
    r"""Encoder-forecaster architecture.

    See :class: `models.convrnn.Encoder` and :class:`models.convrnn.forecaster` for details.
    """

    def __init__(self, forecast_steps, in_channels, out_channels, hidden_channels):
        super(EncoderForecaster, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.forecaster = Forecaster(forecast_steps, out_channels, hidden_channels)
    
    def forward(self, input_):
        states, cells = self.encoder(input_)
        output = self.forecaster(states, cells)
        return output
