# coding: utf8
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

class PropBase(object):

    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    # set the target class as one others as zero. use this vector for back prop
    # def encode_one_hot(self, idx):
    #     one_hot = torch.FloatTensor(1, self.n_class).zero_()
    #     one_hot[0][idx] = 1.0
    #     return one_hot

    # set the target class as one others as zero. use this vector for back prop added by Lezi
    def encode_one_hot_batch(self, z, mu, logvar, mu_avg, logvar_avg):
        one_hot_batch = torch.FloatTensor(z.size()).zero_()
        return mu

    def forward(self, x):
        # self.preds = self.model(x)
        self.image_size = x.size(-1)
        recon_batch, self.mu, self.logvar = self.model(x)
        return recon_batch, self.mu, self.logvar

    # back prop the one_hot signal
    def backward(self, mu, logvar, mu_avg, logvar_avg):
        self.model.zero_grad()
        z = self.model.reparameterize_eval(mu, logvar).to(self.device)
        one_hot = self.encode_one_hot_batch(z, mu, logvar, mu_avg, logvar_avg)

        one_hot = one_hot.to(self.device)
        flag = 2
        if flag == 1:
            self.score_fc = torch.sum(F.relu(one_hot * mu))
        else:
            self.score_fc = torch.sum(one_hot)
        self.score_fc.backward(retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))

class GradCAM(PropBase):

    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads.squeeze())
        self.map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(self.map_size)(self.grads)

    def generate(self):
        # get gradient
        self.grads = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)
        # compute weithts based on the gradient
        self.compute_gradient_weights()

        # get activation
        self.activation = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)

        with torch.no_grad():
            self.activation = self.activation[None, :, :, :, :]
            self.weights = self.weights[:, None, :, :, :]
            gcam = F.conv3d(self.activation, (self.weights.to(self.device)), padding=0, groups=len(self.weights))
            gcam = gcam.squeeze(dim=0)
            gcam = F.upsample(gcam, (self.image_size, self.image_size), mode="bilinear")
            gcam = torch.abs(gcam)

        return gcam

### Save attention maps  ###
def save_cam(image, filename, gcam):
    import numpy as np
    import cv2
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    cv2.imwrite(filename, gcam)