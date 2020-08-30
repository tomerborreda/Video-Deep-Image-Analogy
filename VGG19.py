import setting

import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy


class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x):
        list = []
        for module in self._modules:
            x = self._modules[module](x)
            list.append(x)
        return list


class VGG19:
    def __init__(self, device):

        self.device = device

        url = "https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth"
        vgg19_model = models.vgg19(pretrained=False)
        vgg19_model.load_state_dict(model_zoo.load_url(url), strict=False)
        self.cnn_temp = vgg19_model.features
        self.model = FeatureExtractor()  # the new Feature extractor module network
        conv_counter = 1
        relu_counter = 1
        batn_counter = 1

        block_counter = 1

        for i, layer in enumerate(list(self.cnn_temp)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_layer(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter) + "__" + str(i)
                relu_counter += 1
                self.model.add_layer(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                batn_counter = relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_layer(name, nn.MaxPool2d((2, 2), ceil_mode=True))  # ***

            if isinstance(layer, nn.BatchNorm2d):
                name = "batn_" + str(block_counter) + "_" + str(batn_counter) + "__" + str(i)
                batn_counter += 1
                self.model.add_layer(name, layer)  # ***

        self.model.to(device)
        self.mean_ = (103.939, 116.779, 123.68)

    def forward_subnet(self, input_tensor, start_layer, end_layer):
        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                input_tensor = layer(input_tensor)
        return input_tensor

    def get_features(self, img_tensor, layers):
        img_tensor = img_tensor.to(self.device)

        # assert torch.max(img_tensor)<=1.0 and torch.min(img_tensor)>=0.0, 'inccorect range of tensor'
        for chn in range(3):
            img_tensor[:, chn, :, :] -= self.mean_[chn]

        features_raw = self.model(img_tensor)
        features = []
        for i, f in enumerate(features_raw):
            if (i) in layers:
                features.append(f.detach())
        features.reverse()
        features.append(img_tensor.detach())

        sizes = [f.size() for f in features]
        return features, sizes

    def get_deconvoluted_feat(self, feat, curr_layer, init=None, lr=10, iters=13, display=False):

        blob_layers = [29, 20, 11, 6, 1, -1]
        end_layer = blob_layers[curr_layer]
        mid_layer = blob_layers[curr_layer + 1]
        start_layer = blob_layers[curr_layer + 2] + 1
        if display:
            print(start_layer, mid_layer, end_layer)

        layers = []
        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                if display:
                    print(layer)
                l = copy.deepcopy(layer)
                for p in l.parameters():
                    p.data = p.data.type(torch.cuda.DoubleTensor)
                layers.append(l)
        net = nn.Sequential(*layers)
        noise = init.type(torch.cuda.DoubleTensor).clone()
        target = Variable(feat.type(torch.cuda.DoubleTensor), requires_grad=False)
        noise_size = noise.size()

        def go(x):
            x = x.view(noise_size)
            output = net(x)
            se = torch.mean((target - output) ** 2)
            return se

        init_loss = go(noise).item()
        if setting.isAP:
            setting.optLosses_AP[str(5 - curr_layer)].append(init_loss)
        else:
            setting.optLosses_B[str(5 - curr_layer)].append(init_loss)

        noise = Variable(noise.cuda(), requires_grad=True)
        optimizer = torch.optim.Adam([noise], lr=lr, weight_decay=0)

        def closure():
            optimizer.zero_grad()
            output = net(noise)
            loss = torch.mean((target - output) ** 2)
            loss.backward()
            return loss

        for i in range(50):
            loss = optimizer.step(closure)
            if setting.isAP:
                setting.optLosses_AP[str(5 - curr_layer)].append(loss.item())
            else:
                setting.optLosses_B[str(5 - curr_layer)].append(loss.item())

        end_loss = go(noise).item()
        if setting.isAP:
            setting.optLosses_AP[str(5 - curr_layer)].append(end_loss)
            setting.isAP = False
        else:
            setting.optLosses_B[str(5 - curr_layer)].append(end_loss)
            setting.isAP = True

        print('\tend_loss/init_loss: {:.2f}/{:.2f}'.format(end_loss, init_loss))

        noise = noise.type(torch.cuda.FloatTensor)

        out = self.forward_subnet(input_tensor=noise, start_layer=start_layer, end_layer=mid_layer)
        return out.detach()
