from unet_parts import *

import torch.nn.functional as F 

class Unet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear = True):
		super(Unet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.conv = double_conv(n_channels, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 1024)

		self.up1 = up(512+1024, 512, bilinear)
		self.up2 = up(256+512, 256, bilinear)
		self.up3 = up(128+256, 128, bilinear)
		self.up4 = up(64+128, 64, bilinear)

		self.one = one_conv(64, n_classes)

	def forward(self, x):
		conv = self.conv(x)
		x1 = self.down1(conv)
		x2 = self.down2(x1)
		x3 = self.down3(x2)
		x4 = self.down4(x3)
		x = self.up1(x4, x3)
		x = self.up2(x, x2)
		x = self.up3(x, x1)
		x = self.up4(x, conv)
		x = self.one(x)

		return x
