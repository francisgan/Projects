import torch.nn as nn
import torch.nn.functional as F

#ToDO Fill in the __ values
class FCN_5a(nn.Module):

    def __init__(self, n_class):
        # TODO: Skeleton code given for default FCN network. Fill in the blanks with the shapes
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=2)
        self.bnd4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1, dilation=2)
        self.bnd5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1, dilation=2)
        self.bnd6 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=5, stride=2, padding=1, dilation=2, output_padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1, dilation=2, output_padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
        self.bn6 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    #TODO Complete the forward pass
    def forward(self, x):
        x1 = self.bnd1(self.tanh(self.conv1(x)))
        # Complete the forward function for the rest of the encoder
        x2 = self.bnd2(self.tanh(self.conv2(x1)))
        x3 = self.bnd3(self.tanh(self.conv3(x2)))
        x4 = self.bnd4(self.tanh(self.conv4(x3)))
        x5 = self.bnd5(self.tanh(self.conv5(x4)))
        x6 = self.bnd6(self.tanh(self.conv6(x5)))

        y1 = self.bn1(self.tanh(self.deconv1(x6)))
        # Complete the forward function for the rest of the decoder
        y2 = self.bn2(self.tanh(self.deconv2(y1)))
        y3 = self.bn3(self.tanh(self.deconv3(y2)))
        y4 = self.bn4(self.tanh(self.deconv4(y3)))
        y5 = self.bn5(self.tanh(self.deconv5(y4)))
        y6 = self.bn6(self.tanh(self.deconv6(y5)))

        score = self.classifier(y6)

        return score  # size=(N, n_class, H, W)
