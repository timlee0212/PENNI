###################################################################
#   Defined Decomposed Convolution layer
#
#	For ICML 2020 Submission
#   UNFINISHED RESEARCH CODE
#   DO NOT DISTRIBUTE
#
#   Author: XXXXXXXXXXXX
#   Date:   XXXXXXXXXXXX
##################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import namedtuple

class DecomposedConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
				 dilation=1, device=None, bias=True, init='rand', num_basis=2,):
		super(DecomposedConv2D, self).__init__()
		if isinstance(kernel_size, int):    #If input only one kernel size
			kernel_size = [kernel_size, kernel_size]
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.num_basis = num_basis
		self.device = device

		if bias:
			self.bias = nn.Parameter(torch.randn((out_channels, )), requires_grad=True)
		else:
			self.bias = None

		self.basis = nn.Parameter(torch.randn((num_basis, kernel_size[0] * kernel_size[1])), requires_grad=False)
		self.coefs = nn.Parameter(torch.randn((out_channels * in_channels, num_basis)), requires_grad=True)

	def init_decompose_with_pca(self, basis, coefs):
		self.basis = nn.Parameter(torch.tensor(basis.reshape(self.num_basis, self.kernel_size[0] * self.kernel_size[1])), requires_grad=False)
		self.coefs = nn.Parameter(torch.tensor(coefs.reshape(self.out_channels * self.in_channels, self.num_basis)), requires_grad=True)

	def forward(self, x):
		true_weight = torch.mm(self.coefs, self.basis).view((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
		out = F.conv2d(x, true_weight, self.bias, self.stride, self.padding, self.dilation)

		return out

	def extra_repr(self):
		return 'in_channels={}, out_channels={}, num_basis = {}, bias={}'.format(
			self.in_channels, self.out_channels, self.num_basis, self.bias is not None)

	def forward_test(self, x):
		#No speedup, due to the inefficient implementation using pytorch
		#Efficient Implementation
		# 1          1 1 1 1
		# 2   ---->  2 2 2 2
		# 3          3 3 3 3
		basis_kernel = self.basis.view((self.num_basis, 1, self.kernel_size[0], self.kernel_size[1])).repeat((self.in_channels, 1, 1, 1 ))
		w = ((x.shape[2] + self.padding[0]) - self.kernel_size[0]//2) //self.stride[0]
		mid_fm = F.conv2d(x.repeat((1,self.num_basis, 1, 1)), basis_kernel, self.bias, self.stride, self.padding, self.dilation, groups = self.num_basis * self.in_channels)
		out = F.conv2d(mid_fm, self.coefs.view(self.out_channels, self.in_channels * self.num_basis, 1, 1), stride=1, padding=0, dilation=1)
		#Even Slower
		#out = self.coefs.mm(conv2d_depthwise(x.repeat((1,self.num_basis, 1, 1)), basis_kernel, self.bias, self.stride, self.padding, self.dilation)\
		#	.view((-1, w * w))).view((1, self.out_channels, w, w))
	
		return out
