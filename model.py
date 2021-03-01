### pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()


        self.encoder1 = self.Encoder(config.in_channels, 4, kernel_size=3, stride=1, padding=1, batchnorm=False)
        self.encoder2 = self.Encoder(4,  8, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder3 = self.Encoder(8, 16, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder4 = self.Encoder(16, 32, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder5 = self.Encoder(32, 64, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder6 = self.Encoder(64, 128, kernel_size=3, stride=1, padding=1, batchnorm=True)
        
    
        self.pool1    = nn.MaxPool3d(2)    
        self.pool2    = nn.MaxPool3d(2)
        self.pool3    = nn.MaxPool3d((1, 2, 2))
        self.pool4    = nn.MaxPool3d((1, 2, 2))
        self.pool5    = nn.MaxPool3d((1, 2, 2))
        self.pool6    = nn.MaxPool3d((1, 2, 2))


        self.decoder6 = self.Decoder((config.temporal_num//4, config.frame_size[0]//16,config.frame_size[1]//16), 128 , 64, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder5 = self.Decoder((config.temporal_num//4, config.frame_size[0]//8,config.frame_size[1]//8), 64+64 , 32, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder4 = self.Decoder((config.temporal_num//4, config.frame_size[0]//4,config.frame_size[1]//4), 32 + 32 , 16, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder3 = self.Decoder((config.temporal_num//2, config.frame_size[0]//2,config.frame_size[1]//2), 16 + 16 , 8, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder2 = self.Decoder((config.temporal_num, config.frame_size[0],config.frame_size[1]), 8 + 8, 4, kernel_size=3, stride=1, padding=1,  batchnorm=True)
        self.decoder1 = self.Decoder((config.temporal_num, config.frame_size[0],config.frame_size[1]), 4  + 4 , config.out_channels,  kernel_size=3, stride=(1, 1, 1), padding=1,  batchnorm=True)

        self.out_layer = self.Out_Layer(3 + 3, config.out_channels, kernel_size=3,  stride=(1, 1, 1), padding=1)



    def Encoder(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm, bias=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm3d(out_channels),
                    nn.Tanh()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.Tanh()
            )
        
        return layer
    
    def Decoder(self, size, in_channels, out_channels, kernel_size, stride, padding, batchnorm, bias=False):
        if batchnorm:
            layer = nn.Sequential(
                    nn.Upsample(size=size),
                    nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm3d(out_channels),
                    nn.Tanh()
                    )
        else:
            layer = nn.Sequential(
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.Tanh()
                    )
        
        return layer
    
    def Out_Layer(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        layer =  nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                nn.Tanh()
            )

        return layer


    def forward(self, x):

        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        del p1

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        del p2

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        del p3

        e5 = self.encoder5(p4)
        p5 = self.pool5(e5)
        del p4

        e6 = self.encoder6(p5)
        p6 = self.pool6(e6)
        del p5

        d5 = torch.cat((self.decoder6(p6), e5), 1)
        del e5, p6 

        d4 = torch.cat((self.decoder5(d5), e4), 1)
        del e4, d5 

        d3 = torch.cat((self.decoder4(d4), e3), 1)
        del e3, d4 
        
        d2 = torch.cat((self.decoder3(d3), e2), 1)
        del e2, d3 

        d1 = torch.cat((self.decoder2(d2), e1), 1)
        del e1, d2
        
        out= torch.cat((self.decoder1(d1), x), 1)
        del d1

        out = self.out_layer(out)

        y = out[:, :, 2, :, :]
        del out
        
        return y



class Model_single(nn.Module):
    def __init__(self, config):
        super(Model_single, self).__init__()

        self.encoder1 = self.Encoder(config.in_channels, 4, kernel_size=3, stride=1, padding=1, batchnorm=False)
        self.encoder2 = self.Encoder(4,  8, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder3 = self.Encoder(8, 16, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder4 = self.Encoder(16, 32, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder5 = self.Encoder(32, 64, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.encoder6 = self.Encoder(64, 128, kernel_size=3, stride=1, padding=1, batchnorm=True)
        
    
        self.pool1    = nn.MaxPool2d(2)    
        self.pool2    = nn.MaxPool2d(2)
        self.pool3    = nn.MaxPool2d((2, 2))
        self.pool4    = nn.MaxPool2d((2, 2))
        self.pool5    = nn.MaxPool2d((2, 2))
        self.pool6    = nn.MaxPool2d((2, 2))


        self.decoder6 = self.Decoder((config.frame_size[0]//16,config.frame_size[1]//16), 128 , 64, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder5 = self.Decoder((config.frame_size[0]//8,config.frame_size[1]//8), 64+64 , 32, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder4 = self.Decoder((config.frame_size[0]//4,config.frame_size[1]//4), 32 + 32 , 16, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder3 = self.Decoder((config.frame_size[0]//2,config.frame_size[1]//2), 16 + 16 , 8, kernel_size=3, stride=1, padding=1, batchnorm=True)
        self.decoder2 = self.Decoder((config.frame_size[0],config.frame_size[1]), 8 + 8, 4, kernel_size=3, stride=1, padding=1,  batchnorm=True)
        self.decoder1 = self.Decoder((config.frame_size[0],config.frame_size[1]), 4  + 4 , config.out_channels,  kernel_size=3, stride=(1, 1), padding=1,  batchnorm=True)

        self.out_layer = self.Out_Layer(3 + 3, config.out_channels, kernel_size=3,  stride=(1, 1), padding=1)

    def Encoder(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm, bias=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm2d(out_channels),
                    nn.Tanh()
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.Tanh()
            )
        
        return layer
    
    def Decoder(self, size, in_channels, out_channels, kernel_size, stride, padding, batchnorm, bias=False):
        if batchnorm:
            layer = nn.Sequential(
                    nn.Upsample(size=size),
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm2d(out_channels),
                    nn.Tanh()
                    )
        else:
            layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.Tanh()
                    )
        
        return layer
    
    def Out_Layer(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        layer =  nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.Tanh()
            )

        return layer


    def forward(self, x):

        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        del p1

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        del p2

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        del p3

        e5 = self.encoder5(p4)
        p5 = self.pool5(e5)
        del p4

        e6 = self.encoder6(p5)
        p6 = self.pool6(e6)
        del p5

        d5 = torch.cat((self.decoder6(p6), e5), 1)
        del e5, p6 

        d4 = torch.cat((self.decoder5(d5), e4), 1)
        del e4, d5 

        d3 = torch.cat((self.decoder4(d4), e3), 1)
        del e3, d4 
        
        d2 = torch.cat((self.decoder3(d3), e2), 1)
        del e2, d3 

        d1 = torch.cat((self.decoder2(d2), e1), 1)
        del e1, d2
        
        out= torch.cat((self.decoder1(d1), x), 1)
        del d1

        out = self.out_layer(out)

        y = out[:, :, :, :]
        del out
        
        return y








if __name__ == "__main__":
    from config import Config

    config = Config()
    model = Model_single(config)

    input = torch.randn((1, 3, 360, 640))

    output = model(input)

    print(output.size())


    
