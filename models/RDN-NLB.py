#Residual Dense Network
import torch 
import torch.nn as nn
import torch.nn.init as init

import math

import torch.nn.functional as F

from torch.autograd import Variable



class RDB_Conv(nn.Module):
    def __init__(self, in_C, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_C, growRate, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(in_C),
        #     Modulecell(in_channels=in_C,out_channels=growRate,kernel_size=kSize))
        # self.conv = nn.Sequential(*[
        #     nn.Conv2d(in_C, growRate, kSize, padding=(kSize-1)//2, stride=1),
        #     nn.BatchNorm2d(growRate),
        #     nn.PReLU()
        # ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)
        
class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        
        G0 = growRate0
        G = growRate
        C = nConvLayers
        
        convs = []
        for i in range(C):
            convs.append(RDB_Conv(G0 + i*G, G, kSize))
        self.convs = nn.Sequential(*convs)
        
        #Local Feature Fusion
        self.lff = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        
    def forward(self, x):
        y = self.lff(self.convs(x)) + x
        return y

# add NonLocalBlock2D
# reference: https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
       
        f_div_C = F.softmax(f, dim=1)
        
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0,2,1).contiguous()
         
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z



## define nonlocal mask branch
class NLMaskBranchDownUp(nn.Module):
    def __init__(
        self,growRate0, growRate, nConvLayers, kSize=3,nRDBs):

        super(NLMaskBranchDownUp, self).__init__()

	G0 = growRate0
        G = growRate
        C = nConvLayers
`	m = nRDBs/4
        
        MB_RB1 = []
        MB_RB1.append(NonLocalBlock2D(G0, G0))
	for i in raange(m):
	    MB_RB1.append(RDB(G0, G, C, kSize))
        
        MB_Down = []
        MB_Down.append(nn.Conv2d(G0,G0, 3, stride=2, padding=1))
        
        MB_RB2 = []
        for i in range(2*m):
            MB_RB2.append(RDB(G0, G, C, kSize))
         
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(G0,G0, 6, stride=2, padding=2))   
        
        MB_RB3 = []
        for i in raange(m):
	    MB_RB3.append(RDB(G0, G, C, kSize))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(G0,G0, 1, padding=0, bias=True))
        
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx

        
class RDN(nn.Module):
    def __init__(self, growRate0, growRate, RDBkSize,nConvLayers,nRDBs):
        super(RDN, self).__init__()
        
        G0 = growRate0
        kSize = RDBkSize
	G = growRate
	C = nConvLayers
	D = nRDBs
        
        #D, C, G = (20, 6, 32)
        #D, C, G = (16, 8, 64)
        
        self.RDB_num = D
        
        #Shallow Feature Extraction
        self.sfe1 = nn.Conv2d(3, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.sfe2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        	
        #Residual Dense Blocks
        self.RDBs = nn.ModuleList()
        for i in range(D):
            self.RDBs.append(RDB(G0, G, C, kSize))
	
	self.Mask = []
        self.Mask.append(NLMaskBranchDownUp(G0, G, C, kSize=3,nRDBs))
            
        #Global Feature Fusion
        self.gff = nn.Sequential(*[
            nn.Conv2d(D*G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        #self.non_local = Self_Att(G0, 'relu')

        self.out_conv = nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
    def forward(self, x):
        f1 = self.sfe1(x)
        y = self.sfe2(f1)

	f2= self.Mask(y)
        
        RDB_out = []
        for i in range(self.RDB_num):
            y = self.RDBs[i](y)
            RDB_out.append(y)
            
        y = self.gff(torch.cat(RDB_out, 1))

        #y = self.non_local(y)
        
        y = self.out_conv(f1 + y)
	
	y = y*f2
        y += x
        
        return y
            
        
                    
        

