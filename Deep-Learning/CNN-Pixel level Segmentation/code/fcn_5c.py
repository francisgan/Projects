import torch.nn as nn

#ToDO Fill in the __ values
class FCN_5c(nn.Module):

    def __init__(self, n_class):
        # TODO: Skeleton code given for default FCN network. Fill in the blanks with the shapes
        super().__init__()
        self.n_class = n_class
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation=1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=2, dilation=1)


        self.deconv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.deconv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=2, dilation=1)
        self.deconv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=1)
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=1)
        self.deconv5 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, dilation=1)
        self.deconv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=1)
        self.deconv7 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.deconv8 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=2, dilation=1)
        self.deconv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=1)
        self.deconv10 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.deconv11 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2, dilation=1)
        self.deconv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=1)

        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn1024 = nn.BatchNorm2d(1024)
    #TODO Complete the forward pass
    def forward(self, x):
        x1 = self.bn64(self.relu(self.conv1(x)))
        x2 = self.bn64(self.relu(self.conv2(x1)))
        x3 = self.bn64(self.maxpool(x2))
        x4 = self.bn128(self.relu(self.conv3(x3)))
        x5 = self.bn128(self.relu(self.conv4(x4)))
        x6 = self.bn128(self.maxpool(x5))
        x7 = self.bn256(self.relu(self.conv5(x6)))
        x8 = self.bn256(self.relu(self.conv6(x7)))
        x9 = self.bn256(self.maxpool(x8))
        x10 = self.bn512(self.relu(self.conv7(x9)))
        x11 = self.bn512(self.relu(self.conv8(x10)))
        x12 = self.bn512(self.maxpool(x11))
        x13 = self.bn1024(self.relu(self.conv9(x12)))
        x14 = self.bn1024(self.relu(self.conv10(x13)))

        y1 = self.bn1024(self.relu(self.deconv1(x14)))
        y2 = self.bn512(self.relu(self.deconv2(y1)))
        y3 = self.bn512(self.relu(self.deconv3(y2)))
        y4 = self.bn512(self.relu(self.deconv4(y3)))
        y5 = self.bn256(self.relu(self.deconv5(y4)))
        y6 = self.bn256(self.relu(self.deconv6(y5)))
        y7 = self.bn256(self.relu(self.deconv7(y6)))
        y8 = self.bn128(self.relu(self.deconv8(y7)))
        y9 = self.bn128(self.relu(self.deconv9(y8)))
        y10 = self.bn128(self.relu(self.deconv10(y9)))
        y11 = self.bn64(self.relu(self.deconv11(y10)))
        y12 = self.bn64(self.relu(self.deconv12(y11)))

        score = self.classifier(y12)

        return score  # size=(N, n_class, H, W)
