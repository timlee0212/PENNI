import numpy as np
import torch.nn as nn
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from decompose import decomConv

class param_resolver:
    def __init__(self, model, quant=False):
        self.model = model
        self.layer_index = {}
        self.num_layers = 0
        #self.current_gpu = config.current_gpu
        self.params = []
        self.params_normed = []
        self._store_params(model.named_parameters())

    def _store_params(self, named_parameters):
        layer_idx = 0
        flt_3d = 0
        flt_2d = 0

        for name, param in named_parameters:

            if 'weight' in name and len(param.shape)==4:
                if param.shape[2] == 1:
                    print("Skipping 1x1 Conv...")
                    continue

                self.layer_index[layer_idx] = name

                param = param.cpu().detach()
                # if self.quant:
                #     param = param.astype(dtype=int) #Using int to evaluate accurately

                self.params.append(param.numpy())
                layer_idx += 1
                flt_3d += param.shape[0]
                flt_2d += param.shape[0] * param.shape[1]

                print("Layer:%s ---- \t %d x %d Filters with shape (%d, %d)"
                        % (name, param.shape[0], param.shape[1], param.shape[2], param.shape[3]))

        print("%d Conv Layers Loaded, have %d 3D filters and %d 2D kernels in total." % ( layer_idx+1, flt_3d, flt_2d))
        self.num_layers = layer_idx
        self.params_normed = np.array(self.params)
        #self._normilize_weight()

    #Normalized each layer
    def _normilize_weight(self, norm='l2'):
        self.coef = []
        self.params_normed = copy.deepcopy(self.params)
        for lidx in range(self.num_layers):
            #Skip 1x1 Conv
            if self.params_normed[lidx].shape[2]==1:
                print("Skipping 1x1 Conv...")
                self.coef.append(1)
                continue
            #Store Normalization Coefficient
            self.coef.append(np.zeros(self.params_normed[lidx].shape[:2]))
            for i in range(self.params_normed[lidx].shape[1]):
                for o in range(self.params_normed[lidx].shape[0]):
                    filter = self.params_normed[lidx][o, i, :, :]
                    if norm == 'l2':
                        coef = np.sqrt(np.sum(filter ** 2))
                    elif norm == 'l1':
                        coef = np.sum(np.abs(filter))
                    elif norm == 'l0':
                        coef = np.sum(filter!=0)
                    else:
                        raise NotImplementedError("Not Supported Norm.")
                    #Fix Sparse Situation
                    self.params_normed[lidx][o, i, :, :] = (filter / coef) if coef!=0 else filter
                    self.coef[lidx][o, i] = coef

    def PCA_decomposing(self, basis=2, layers=None):
        if layers==None:
            layers = np.arange(self.num_layers)

        if not isinstance(basis, list):
            basis = [basis] * len(layers)

        error_list = []

        for lidx in layers:

            layer_name = self.layer_index[lidx].split('.')

            print("Decomposing Layer:", self.layer_index[lidx], " with", basis[lidx], "Basis Filters")
            in_channel = self.params_normed[lidx].shape[1]
            out_channel = self.params_normed[lidx].shape[0]
            num_filters = in_channel * out_channel
            filter_size = self.params_normed[lidx].shape[2]

            decomposer = PCA(n_components=basis[lidx])
            weight = self.params_normed[lidx].reshape(in_channel*out_channel, filter_size**2)
            decom_coef = decomposer.fit_transform(weight)
            decom_basis = decomposer.components_
            decom_bias = decomposer.mean_

            c = np.matmul(decom_coef, decom_basis) + decom_bias
            c = c.reshape(out_channel, in_channel, filter_size, filter_size)

            error_list.append((c - self.params_normed[lidx]).flatten())

            error = 0
            for o in range(out_channel):
                for i in range(in_channel):
                    error += np.sqrt(np.average((c[o, i, :, :] - self.params_normed[lidx][o, i, :, :])**2))
            error = error/(out_channel+in_channel)
            print("Decomposing Error:", error)

            #NEW VERSION - Recursively Replace Conv Models
            parent = self.model
            for mkey in layer_name:
                n_parent = parent._modules[mkey]
                if len(n_parent._modules) == 0 and isinstance(n_parent, nn.Conv2d):     #Is a basic operation
                    print(mkey)
                    ori_conv = n_parent
                    parent._modules[mkey] = decomConv.DecomposedConv2D(ori_conv.in_channels, ori_conv.out_channels,
                                        ori_conv.kernel_size, ori_conv.stride, num_basis=basis[lidx] ,
                                       padding=ori_conv.padding, dilation=ori_conv.dilation, bias=ori_conv.bias, device='cuda')
                    parent._modules[mkey].init_decompose_with_pca(decom_basis, decom_coef)
                    break
                else:
                    parent = n_parent

        return self.model

    def plot_params_dist(self, dim=2, method='pca', layer_idx= None):
        assert dim in [2, 3], "Must be 2D or 3D."
        total_dist = np.empty((0, dim))
        total_y = np.array([])
        if layer_idx is None:
            layer_idx = np.arange(self.num_layers)

        for lidx in layer_idx:
            if self.params[lidx].shape[2] == 1:
                print("Drawing scheme: Layer: %s, Skipping 1x1 Conv...")
                continue

            filters = self.params[lidx].reshape(-1, self.params[lidx].shape[2] * self.params[lidx].shape[3])
            if method == 'pca':
                decomposer = PCA(n_components = dim)
            elif method == 'tsne':
                decomposer = TSNE(n_components = dim, perplexity=10)
            elif method=='mds':
                decomposer = MDS(n_components = dim)
            else:
                pass
            dist = decomposer.fit_transform(filters)
            total_dist = np.vstack((total_dist, dist))
            y = np.repeat(lidx, dist.shape[0])
            total_y = np.hstack((total_y, y))

            if dim==2:
                plt.scatter(dist[:, 0], dist[:, 1])
            else:
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(dist[:, 0], dist[:, 1], dist[:, 2])
            plt.title(self.layer_index[lidx])
            plt.savefig("./%s.jpg"%lidx)
            plt.show()

    def _cal_mse(self, basis, ori_filters, coefs):
        error = 0
        error_ori = 0
        error_elt = np.zeros(basis.shape[1:])
        error_elt_ori = np.zeros(basis.shape)
        for idx in range(ori_filters.shape[0]):
            error_item = np.square(np.abs(ori_filters[idx] - basis))
            error_item_ori = np.square(np.abs(ori_filters[idx] * coefs[idx] - basis * coefs[idx]))
            error_elt += error_item
            error += np.sum(error_item)
            error_elt_ori += error_item_ori
            error_ori += np.sum(error_item_ori)

        error_elt = np.sqrt(error_elt / ori_filters.shape[0])
        error = np.sqrt(error / ori_filters.shape[0])

        error_elt_ori = np.sqrt(error_elt_ori / ori_filters.shape[0])
        error_ori = np.sqrt(error_ori / ori_filters.shape[0])

        return error, error_elt, error_ori, error_elt_ori


