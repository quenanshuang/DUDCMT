import torch
import torch.nn as nn
from models.networkLib.segmodels import deeplabv3_resnet50, deeplabv3p_resnet50
from timm.models.layers import trunc_normal_
from models.DSCnet import DSConv1,DSC1,EncoderConv,DecoderConv,DSCNet
from models.Dcoonet import DconnNet
from models.sanet import SA_UNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
class DeepLabv3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=False):
        super(DeepLabv3, self).__init__()
        self.net = deeplabv3_resnet50(in_channels=in_channels, num_classes=out_channels, pretrained=pretrained)

    def forward(self, x):
        x = self.net(x)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)


class DeepLabv3p(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=False):
        super(DeepLabv3p, self).__init__()
        self.net = deeplabv3p_resnet50(in_channels=in_channels, num_classes=out_channels, pretrained=pretrained)

    def forward(self, x):
        x = self.net(x)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)

    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

############################DSCunet
class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

class Encoder1(nn.Module):
    def __init__(self, params):
        super(Encoder1, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.in_conv_0x = DSConv(
            self.in_chns,
           self.ft_chns[0],
            kernel_size=9,
            extend_scope=1.0,
            morph=0,
            if_offset=True,
            device="cuda:3",
        )
        self.in_conv_0y = DSConv(
            self.in_chns,
           self.ft_chns[0],
            kernel_size=9,
            extend_scope=1.0,
            morph=1,
            if_offset=True,
            device="cuda:3",
        )
        self.conv1x1_0=nn.Conv2d(
            self.ft_chns[0]*3, self.ft_chns[0], kernel_size=1, padding=0, bias=True)
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down1x = DSConv(
            self.ft_chns[1],
           self.ft_chns[1],
            kernel_size=9,
            extend_scope=1.0,
            morph=0,
            if_offset=True,
            device="cuda:3",
        )
        self.down1y = DSConv(
            self.ft_chns[1],
           self.ft_chns[1],
            kernel_size=9,
            extend_scope=1.0,
            morph=1,
            if_offset=True,
            device="cuda:3",
        )
        self.conv1x1_1=nn.Conv2d(
            self.ft_chns[1]*3, self.ft_chns[1], kernel_size=1, padding=0, bias=True)
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down2x = DSConv(
            self.ft_chns[2],
           self.ft_chns[2],
            kernel_size=9,
            extend_scope=1.0,
            morph=0,
            if_offset=True,
            device="cuda:3",
        )
        self.down2y = DSConv(
            self.ft_chns[2],
           self.ft_chns[2],
            kernel_size=9,
            extend_scope=1.0,
            morph=1,
            if_offset=True,
            device="cuda:3",
        )
        self.conv1x1_2=nn.Conv2d(
            self.ft_chns[2]*3, self.ft_chns[2], kernel_size=1, padding=0, bias=True)
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down3x = DSConv(
            self.ft_chns[3],
           self.ft_chns[3],
            kernel_size=9,
            extend_scope=1.0,
            morph=0,
            if_offset=True,
            device="cuda:3",
        )
        self.down3y = DSConv(
            self.ft_chns[3],
           self.ft_chns[3],
            kernel_size=9,
            extend_scope=1.0,
            morph=1,
            if_offset=True,
            device="cuda:3",
        )
        self.conv1x1_3=nn.Conv2d(
            self.ft_chns[3]*3, self.ft_chns[3], kernel_size=1, padding=0, bias=True)
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.down4x = DSConv(
            self.ft_chns[4],
           self.ft_chns[4],
            kernel_size=9,
            extend_scope=1.0,
            morph=0,
            if_offset=True,
            device="cuda:3",
        )
        self.down4y = DSConv(
            self.ft_chns[4],
           self.ft_chns[4],
            kernel_size=9,
            extend_scope=1.0,
            morph=1,
            if_offset=True,
            device="cuda:3",
        )
        self.conv1x1_4=nn.Conv2d(
            self.ft_chns[4]*3, self.ft_chns[4], kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        x01 = self.in_conv(x)
        x0_x = self.in_conv_0x(x)
        x0_y = self.in_conv_0y(x)
        x0 = self.conv1x1_0(torch.cat([x01,x0_x,x0_y],dim=1))
        # print("x0",x0.size())
        x11 = self.down1(x0)
        # print("x11",x11.size())
        x1_x = self.down1x(x11)
        # print("x1x",x1_x.size())
        x1_y = self.down1y(x11)
        # print("x1y",x1_y.size())
        x1 = self.conv1x1_1(torch.cat([x11,x1_x,x1_y],dim=1))
        x21 = self.down2(x1)
        x2_x = self.down2x(x21)
        x2_y = self.down2y(x21)
        x2 = self.conv1x1_2(torch.cat([x21,x2_x,x2_y],dim=1))
        x31 = self.down3(x2)
        x3_x = self.down3x(x31)
        x3_y = self.down3y(x31)
        x3 = self.conv1x1_3(torch.cat([x31,x3_x,x3_y],dim=1))
        x41 = self.down4(x3)
        x4_x = self.down4x(x41)
        x4_y = self.down4y(x41)
        x4 = self.conv1x1_4(torch.cat([x41,x4_x,x4_y],dim=1))
        return [x0, x1, x2, x3, x4]
class UNetorg1(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNetorg1, self).__init__()

        params1 = {'in_chns': in_channels,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': num_classes,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.backbone = Encoder1(params1)
        self.classifier = Decoder(params1)
        self.kaiming_normal_init_weight()

    def forward(self, x):
        feature = self.backbone(x)
        output = self.classifier(feature)
        return {'out': output}
    
    def kaiming_normal_init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class UNet_dsc(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet_dsc, self).__init__()
        self.net = UNetorg1(in_channels=in_channels, num_classes=out_channels)

    def forward(self, x):
        x = self.net(x)
        return x
    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)
###########################################################
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

    
class UNetorg(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNetorg, self).__init__()

        params1 = {'in_chns': in_channels,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': num_classes,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.backbone = Encoder(params1)
        self.classifier = Decoder(params1)
        self.kaiming_normal_init_weight()

    def forward(self, x):
        feature = self.backbone(x)
        output = self.classifier(feature)
        return {'out': output}
    
    def kaiming_normal_init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.net = UNetorg(in_channels=in_channels, num_classes=out_channels)

    def forward(self, x):
        x = self.net(x)
        return x

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)

########################frunet
class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = x
        x = self.conv(x)
        out = x + res
        out = self.relu(out)
        return x


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1+x2+x3)
        return out


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class block(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dp=dp)
        if self.is_up == True:
            self.up = up(out_c, out_c//2)
        if self.is_down == True:
            self.down = down(out_c, out_c*2)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.neg_slope)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
class FR_UNet(nn.Module):
    def __init__(self,  num_classes=2, num_channels=3, feature_scale=2,  dropout=0.2, fuse=True, out_ave=True):
        super(FR_UNet, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = block(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block10 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block11 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block12 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block13 = block(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block2_2 = block(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = block(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block22 = block(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block3_1 = block(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block30 = block(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block31 = block(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block40 = block(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
            
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(InitWeights_He)

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))
        # print("x13的size")
        # print(x13.shape)
        # print("x1_2的size")
        # print(x1_2.shape)
        # print(x1_1.size())
        # print(x10.size())
        # print(x11.size())
        # print(x12.size())
        # print(x13.size())
        if self.out_ave == True:
            output = (self.final1(x1_1)+self.final2(x10) +
                      self.final3(x11)+self.final4(x12)+self.final5(x13))/5
        else:
            output = self.final5(x13)
        # output=torch.cat([self.final3(x11), self.final5(x13)], dim=1)
        # return x1_2,output,x13
        return output

class FR_UNet1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FR_UNet1, self).__init__()
        self.net = FR_UNet(num_classes=out_channels,num_channels=in_channels )

    def forward(self, x):
        x = self.net(x)
        return {'out': x}

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)

class DSC_NET1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,device2=device1):
        super(DSC_NET1, self).__init__()
        self.net = DSCNet(n_channels=in_channels,n_classes=out_channels,device=device2 )

    def forward(self, x):
        x = self.net(x)
        return {'out': x}

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)

class Dcoonet1(nn.Module):
    def __init__(self,  out_channels=1):
        super(Dcoonet1, self).__init__()
        self.net = DconnNet(num_class=out_channels )

    def forward(self, x):
        x,x2 = self.net(x)
        return {'out': x,'out2':x2}

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)

class SA_UNet1(nn.Module):
    def __init__(self,  in_channels=1,out_channels=1):
        super(SA_UNet1, self).__init__()
        self.net = SA_UNet(in_channels=in_channels,num_classes=out_channels )

    def forward(self, x):
        x = self.net(x)
        return {'out': x}

    def detach_model(self):
        for param in self.parameters():
            param.detach_()

    def ema_update(self, student, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, s_param in zip(self.parameters(), student.parameters()):
            t_param.data.mul_(ema_decay).add_(1 - ema_decay, s_param.data)
    
    def weighted_update(self, teacher1, teacher2, coefficient=0.99, ema_decay=0.99, cur_step=None):
        if cur_step is not None:
            ema_decay = min(1 - 1 / (cur_step + 1), ema_decay)
        for t_param, t1_param, t2_param in zip(self.parameters(), teacher1.parameters(), teacher2.parameters()):
            t_param.data.mul_(ema_decay).add_(coefficient*(1-ema_decay), t1_param.data).add_((1-coefficient)*(1-ema_decay), t2_param.data)


if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([2, 1, 256, 256]).to(device)
    deeplab = DeepLabv3p(in_channels=1, out_channels=4, pretrained=True).to(device)
    result = deeplab(tensor)
    print('#parameters:', sum(param.numel() for param in deeplab.parameters()))
    print(result['out'].shape)