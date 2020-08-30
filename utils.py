import setting

import torch
import cv2
import numpy as np

def ts2np(x):
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = x.transpose(1,2,0)
    return x

def np2ts(x, device):
    x = x.transpose(2,0,1)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.to(device)
    return x


def blend(response, f_a, r_bp, alpha=0.8, tau=0.05):
    """
    :param response:
    :param f_a: feature map (either F_A or F_BP)
    :param r_bp: reconstructed feature (R_BP or R_A)
    :param alpha: scalar balance the ratio of content and style in new feature map
    :param tau: threshold, default: 0.05 (suggested in paper)
    :return: (f_a*W + r_bp*(1-W)) where W=alpha*(response>tau)

    Following the official implementation, I replace the sigmoid function (stated in paper) with indicator function
    """
    weight = (response > tau).type(f_a.type()) * alpha
    weight = weight.expand(1, f_a.size(1), weight.size(2), weight.size(3))

    f_ap = f_a*weight + r_bp*(1. - weight)
    return f_ap


def normalize(feature_map):
    """
    :param feature_map: either F_a or F_bp
    :return:
    normalized feature map
    response
    """
    response = torch.sum(feature_map*feature_map, dim=1, keepdim=True)
    normed_feature_map = feature_map/torch.sqrt(response)

    # response should be scaled to (0, 1)
    response = (response-torch.min(response))/(torch.max(response)-torch.min(response))
    return  normed_feature_map, response


def load_image(file_A, resizeRatio=1.0):
    ori_AL = cv2.imread(file_A)
    ori_img_sizes = ori_AL.shape[:2]

    # resize
    if (ori_AL.shape[0] > 700):
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[1] > 700):
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[0] < 200):
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    if (ori_AL.shape[1] < 200):
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[0]*ori_AL.shape[1] > 350000):
        ratio = np.sqrt(350000 / (ori_AL.shape[1]*ori_AL.shape[0]))
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    img_A = cv2.resize(ori_AL,None, fx=resizeRatio, fy=resizeRatio, interpolation=cv2.INTER_CUBIC)

    return img_A


def changes_from_prev_frame(curr, prev):
    # Returns a boolean 2D matrix with the images shape.
    curr = np.array(curr)
    prev = np.array(prev)

    y_size = curr.shape[0]
    x_size = curr.shape[1]

    is_prev_frame_changed = np.ones((y_size, x_size), dtype=bool)

    for ay in range(y_size):
        for ax in range(x_size):
            if (curr[ay, ax, 0] >= prev[ay, ax, 0]-2) and (curr[ay, ax, 0] <= prev[ay, ax, 0]+2) \
                and (curr[ay, ax, 1] >= prev[ay, ax, 1]-2) and (curr[ay, ax, 1] <= prev[ay, ax, 1]+2) \
                and (curr[ay, ax, 2] >= prev[ay, ax, 2]-2) and (curr[ay, ax, 2] <= prev[ay, ax, 2]+2):
                is_prev_frame_changed[ay, ax] = False

    return is_prev_frame_changed


def plot_stages_loss_and_save(losses, path_name):
    import pandas as pd
    import seaborn as sns

    data = {'Time': [], 'Loss': [], 'Stage': []}

    for stage in ['5', '4', '3', '2']:
        if stage == '5':
            data['Time'].extend(list(range(0, len(losses[stage]))))
        else:
            data['Time'].extend(list(range(data['Time'][-1]+1, data['Time'][-1]+len(losses[stage])+1)))
        data['Loss'].extend(losses[stage])
        data['Stage'].extend([stage]*len(losses[stage]))

    df = pd.DataFrame.from_dict(data)

    sns.set_style(style='white')
    sns_plot = sns.relplot(x="Time", y="Loss",
                hue="Stage",
                facet_kws=dict(sharey=False),
                kind="line",
                legend="full",
                palette=sns.color_palette("colorblind", 4),
                data=df);

    sns_plot.savefig(path_name)
