import torch
import torch.nn as nn
import torch.nn.functional as F


class PRLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super(PRLinear, self).__init__(in_features, out_features, bias)
        self.eps = eps

    def forward(self, x):
        # compute the length of w and x. We find this is faster than the norm, although the later is simple.
        w_len = torch.sqrt((torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))).clamp_(min=self.eps))  # 1*num_classes
        x_len = torch.sqrt((x.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # batch*1

        # compute the cosine of theta and abs(sine) of theta.
        wx_len = torch.matmul(x_len, w_len).clamp_(min=self.eps)
        cos_theta = (torch.matmul(x, torch.t(self.weight)) / wx_len).clamp_(-1.0, 1.0)  # batch*num_classes
        abs_sin_theta = torch.sqrt(1.0 - cos_theta ** 2)  # batch*num_classes

        # PR Product
        out = wx_len * (abs_sin_theta.detach() * cos_theta + cos_theta.detach() * (1.0 - abs_sin_theta))

        # to save memory
        del w_len, x_len, wx_len, cos_theta, abs_sin_theta

        if self.bias is not None:
            out = out + self.bias

        return out


class PRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-8):
        super(PRConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        assert groups == 1, 'Currently, we do not realize the PR for group CNN. Maybe you can do it yourself and welcome for pull-request.'

        self.eps = eps
        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))

    def forward(self, input):
        # compute the length of w
        w_len = torch.sqrt((self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).t()).clamp_(
            min=self.eps))  # 1*out_channels

        # compute the length of x at each position with the help of convolutional operation
        x_len = input.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
        x_len = torch.sqrt((F.conv2d(x_len, self.ones_weight, None,
                                     self.stride,
                                     self.padding, self.dilation, self.groups)).clamp_(
            min=self.eps))  # batch*1*H_out*W_out

        # compute the cosine of theta and abs(sine) of theta.
        wx_len = (x_len * (w_len.unsqueeze(-1).unsqueeze(-1))).clamp_(min=self.eps)  # batch*out_channels*H_out*W_out
        cos_theta = (F.conv2d(input, self.weight, None, self.stride,
                              self.padding, self.dilation, self.groups) / wx_len).clamp_(-1.0,
                                                                                         1.0)  # batch*out_channels*H_out*W_out
        abs_sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # PR Product
        out = wx_len * (abs_sin_theta.detach() * cos_theta + cos_theta.detach() * (1.0 - abs_sin_theta))

        # to save memory
        del w_len, x_len, wx_len, cos_theta, abs_sin_theta

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class PRLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(PRLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # replace linear with PRLinear
        self.ih_linear = PRLinear(input_size, 4 * hidden_size, bias)
        self.hh_linear = PRLinear(hidden_size, 4 * hidden_size, bias)

        self.weight_ih = self.ih_linear.weight
        self.bias_ih = self.ih_linear.bias

        self.weight_hh = self.hh_linear.weight
        self.bias_hh = self.hh_linear.bias

    def forward(self, input, hidden):
        hx, cx = hidden

        gates = self.ih_linear(input) + self.hh_linear(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * F.tanh(cy)

        return hy, cy
