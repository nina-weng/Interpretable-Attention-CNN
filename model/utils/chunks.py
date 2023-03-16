import torch.nn as nn
import torch

class CNN2DBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(32,32)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # 2d

        self.pool_kernel_size = 9
        self.pool_dilation = 1
        self.pool_padding = int((self.pool_kernel_size - 1) // 2)

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding='same')
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.leakyrule = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=self.pool_kernel_size, dilation=self.pool_dilation,
                                    padding=self.pool_padding, stride=1)


    def forward(self,x):
        out = self.conv(x)
        out = self.leakyrule(out)
        out = self.bn(out)
        out = self.maxpool(out)

        return out

class ShortCut2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(1,1)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              padding='same')
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)


    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class CNN1DTSBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(32,16)):
        super().__init__()
        self.in_channels = in_channels # (timepoints, num_electrodes)
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # 1d

        self.t_kernel = self.kernel_size[0]
        self.s_kernel = self.kernel_size[1]

        self.pool_kernel_size = 9
        self.pool_dilation = 1
        self.pool_padding = int((self.pool_kernel_size - 1) // 2)

        self.conv_time = nn.Conv1d(in_channels=self.in_channels[0],
                              out_channels=self.out_channels,
                                   kernel_size=self.t_kernel,
                                   padding='same')
        self.conv_spatial = nn.Conv1d(in_channels=self.in_channels[1],
                                   out_channels=self.out_channels,
                                   kernel_size=self.s_kernel,
                                   padding='same')

        self.leakyrule = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=self.pool_kernel_size,
                                         dilation=self.pool_dilation,
                                         padding=self.pool_padding, stride=1)
        self.bn = nn.BatchNorm2d(1)




    def forward(self,x):
        # x : (batch_size, timepoints, num_electrodes)
        out = self.conv_time(x) # (batch_size,out_c, num_electrodes)
        out = torch.permute(out,(0,2,1)) # (batch_size, num_electrodes, out_c)
        out = self.conv_spatial(out)  # (batch_size, out_c, out_c)
        out = torch.permute(out, (0, 2, 1))

        out = self.leakyrule(out)

        out = torch.unsqueeze(out,dim=1) # (batch_size, 1, out_c, out_c)
        out = self.bn(out)
        out = self.maxpool(out) # (batch_size, 1, out_c, out_c)


        out = torch.squeeze(out, dim=1) # (batch_size, out_c, out_c)

        return out


class ShortCut1DTS(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1):
        super().__init__()
        self.in_channels = in_channels # (time,num_elec)
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv_time = nn.Conv1d(in_channels=self.in_channels[0],out_channels=self.out_channels,
                              kernel_size=self.kernel_size,padding='same')
        self.conv_spatial = nn.Conv1d(in_channels=self.in_channels[1], out_channels=self.out_channels,
                                   kernel_size=self.kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(1)


    def forward(self,x):
        out = self.conv_time(x) # (batch_size,out_c, num_electrodes)
        out = torch.permute(out, (0, 2, 1))  # (batch_size, num_electrodes, out_c)
        out = self.conv_spatial(out)  # (batch_size, out_c, out_c)
        out = torch.permute(out, (0, 2, 1))
        out = torch.unsqueeze(out, dim=1)  # (batch_size, 1, out_c, out_c)
        out = self.bn(out)
        out = torch.squeeze(out, dim=1)  # (batch_size, out_c, out_c)
        return out


class CNN1DTBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size #1d

        # self.pool_kernel_size = self.kernel_size if self.kernel_size%2 == 1 else self.kernel_size-1
        self.pool_kernel_size = 9
        self.pool_dilation = 1
        self.pool_padding = int((self.pool_kernel_size-1) //2)

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(num_features = self.out_channels)
        self.leakyrule = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=self.pool_kernel_size,dilation=self.pool_dilation,
                                    padding=self.pool_padding,stride=1)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leakyrule(out)
        out = self.maxpool(out)

        return out

class ShortCut1DT(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channels,
                              kernel_size=self.kernel_size,padding='same')
        self.bn = nn.BatchNorm1d(num_features = self.out_channels)


    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class Classifier(nn.Module):
    def __init__(self,input_features,hidden_para,num_class):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=input_features,out_features=hidden_para)
        self.bn = nn.BatchNorm1d(num_features=hidden_para)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=hidden_para,out_features=num_class)
        self.softmax = nn.Softmax()

    def forward(self,x):
        out = self.linear_1(x)
        if out.shape[0] != 1:
            out = self.bn(out)
        out = self.relu(out)
        out = self.linear_2(out)
        # out = self.softmax(out)

        return out


