import setting

import os
from utils import load_image, plot_stages_loss_and_save
import argparse
from DeepAnalogy import analogy
import cv2
import datetime

def str2bool(v):
    return v.lower() in ('true')

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--LR', type=float, default=[2, 2, 2, 2], nargs='+')
    parser.add_argument('--resize_ratio', type=float, default=0.5)
    parser.add_argument('--weight', type=int, default=2, choices=[2,3])
    parser.add_argument('--img_BP_path', type=str, default='data/husky_style.jpg')
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--video_path', type=str, default='data/husky_frames')

    # Debug mode enable more detailed timers and loss plot
    parser.add_argument('--debug', type=str2bool, default=False)

    args = parser.parse_args()
    
    assert len(args.LR) == 4
    setting.init()
    details = ""
    setting.details = "".join([setting.details, "{}\n".format(args)])

    setting.is_debug_mode = args.debug

    # load images
    img_BP = load_image(args.img_BP_path, args.resize_ratio)

    # setting parameters
    config = dict()

    params = {
        'layers': [29,20,11,6,1],
        'iter': 10,
    }
    config['params'] = params

    if args.weight == 2:
        config['weights'] = [1.0, 0.8, 0.7, 0.6, 0.1, 0.0]
    elif args.weight == 3:
        config['weights'] = [1.0, 0.9, 0.8, 0.7, 0.2, 0.0]
    config['nnf_patch_size'] = [3,3,3,5,5,3]
    config['rangee'] = [32,6,6,4,4,2]

    config['use_cuda'] = args.use_cuda
    config['lr'] = args.LR

    # save result
    content = os.listdir('results')
    count = 1
    for c in content:
        if os.path.isdir('results/' + c):
            count += 1
    save_path = 'results/expr_{}'.format(count)
    os.mkdir(save_path)

    # Deep-Image-Analogy
    print("\n##### Deep Image Analogy On Video - start #####")
    elapse_all = 0

    # number of frames
    frames_list = os.listdir(args.video_path)
    frames_number = len(frames_list)
    print('There are {} frames'.format(frames_number))

    img_A_prev = None
    for i in range(frames_number):
        print("Loading images for frame {}".format(i), end='')
        filename = args.video_path + "/{}.jpg".format(i)
        img_A = load_image(filename, args.resize_ratio)
        print("\n##### Deep Image Analogy - start #####")
        img_AP, elapse = analogy(img_A, img_BP, config)
        elapse_all += elapse
        elapse_str = str(datetime.timedelta(seconds=elapse))[:-7]
        print("##### Deep Image Analogy - end | Elapse:" + elapse_str + " #####")
        if i == 0:
            setting.is_first_frame = False
        setting.prev_frame = img_A

        cv2.imwrite(save_path + '/{}.jpg'.format(i), img_AP)
        print('Image saved!')

    elapse_all_str = str(datetime.timedelta(seconds=elapse_all))[:-7]
    print("##### Deep Image Analogy On Video - end | Elapse:"+elapse_all_str+" #####")

    with open(save_path+"/Details.txt", "w") as text_file:
        print(setting.details, file=text_file)

    if setting.is_debug_mode:
        plot_stages_loss_and_save(setting.optLosses_AP, save_path+"/Losses_AP.png")
        plot_stages_loss_and_save(setting.optLosses_B, save_path+"/Losses_B.png")

    print('Image saved!')
