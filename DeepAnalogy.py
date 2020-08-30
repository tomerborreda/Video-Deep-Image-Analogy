import setting

from VGG19 import VGG19
from PatchMatchOrig import init_nnf, upSample_nnf, avg_vote, propagate, reconstruct_avg
import copy
from utils import *
import time
import datetime


def analogy(img_A, img_BP, config):
    start_time_0 = time.time()

    weights = config['weights']
    nnf_patch_size = config['nnf_patch_size']
    rangee = config['rangee']
    params = config['params']
    lr = config['lr']
    if config['use_cuda']:
        device = torch.device('cuda:0')
    else:
        raise NotImplementedError('cpu mode is not supported yet')

    # preparing data
    img_A_tensor = torch.FloatTensor(img_A.transpose(2, 0, 1))
    img_BP_tensor = torch.FloatTensor(img_BP.transpose(2, 0, 1))
    img_A_tensor, img_BP_tensor = img_A_tensor.to(device), img_BP_tensor.to(device)

    img_A_tensor = img_A_tensor.unsqueeze(0)
    img_BP_tensor = img_BP_tensor.unsqueeze(0)

    # 4.1 Preprocessing Step
    # compute 5 feature maps
    model = VGG19(device=device)
    F_A, F_A_size = model.get_features(img_tensor=img_A_tensor.clone(), layers=params['layers'])
    F_BP, F_B_size = model.get_features(img_tensor=img_BP_tensor.clone(), layers=params['layers'])

    # Init AP&B 's feature maps with F_A&F_BP
    F_AP = copy.deepcopy(F_A)
    F_B = copy.deepcopy(F_BP)

    print("Features extracted!")
    # Note that the feature_maps now is in the order of [5,4,3,2,1,input]

    if not setting.is_first_frame:
        # Check for change from prev image
        setting.is_prev_frame_changed = changes_from_prev_frame(img_A, setting.prev_frame)
        print("% of frame changed from prev one: {}".format(np.sum(setting.is_prev_frame_changed)/(setting.is_prev_frame_changed.shape[0] * setting.is_prev_frame_changed.shape[1])))

    for curr_layer in range(5):
        print("\n### current stage: %d - start ###"%(5-curr_layer))
        start_time_1 = time.time()
        
        setting.currlayer_global = curr_layer
        if curr_layer == 0:
            if not setting.is_first_frame:
                ann_AB = setting.nnf_global["stage 5"][0]
                ann_BA = setting.nnf_global["stage 5"][2]
            else:
                ann_AB = init_nnf(F_A_size[curr_layer][2:], F_B_size[curr_layer][2:])
                ann_BA = init_nnf(F_B_size[curr_layer][2:], F_A_size[curr_layer][2:])
        else:
            ann_AB = upSample_nnf(ann_AB, F_A_size[curr_layer][2:])
            ann_BA = upSample_nnf(ann_BA, F_B_size[curr_layer][2:])

        # Blend feature
        # According to Equotion(2), we need to normalize F_A and F_BP
        # response denotes the M in Equotion(6)
        F_A_BAR, response_A = normalize(F_A[curr_layer])
        F_BP_BAR, response_BP = normalize(F_BP[curr_layer])

        # F_AP&F_B is reconstructed according to Equotion(4)
        # Note that we reuse the varibale F_AP here,
        # it denotes the RBprime as is stated in the  Equotion(4) which is calculated
        # at the end of the previous iteration
        F_AP[curr_layer] = blend(response_A, F_A[curr_layer], F_AP[curr_layer], weights[curr_layer])
        F_B[curr_layer] = blend(response_BP, F_BP[curr_layer], F_B[curr_layer], weights[curr_layer])

        # Normalize F_AP&F_B as well
        F_AP_BAR, _ = normalize(F_AP[curr_layer])
        F_B_BAR, _ = normalize(F_B[curr_layer])

        # NNF search, Run PatchMatch algorithm to get mapping AB and BA
        print("- NNF search for ann_AB")
        start_time_2 = time.time()
        ann_AB, ann_AB_nnd = propagate(ann_AB, ts2np(F_A_BAR), ts2np(F_AP_BAR), ts2np(F_B_BAR), ts2np(F_BP_BAR), nnf_patch_size[curr_layer],
                              params['iter'], rangee[curr_layer])
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])

        # If this is the last iteration then do not search for mapping
        if curr_layer != 4:
            print("- NNF search for ann_BA")
            start_time_2 = time.time()
            ann_BA, ann_BA_nnd = propagate(ann_BA, ts2np(F_BP_BAR), ts2np(F_B_BAR), ts2np(F_AP_BAR), ts2np(F_A_BAR), nnf_patch_size[curr_layer],
                                  params['iter'], rangee[curr_layer])
            print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])
            setting.nnf_global["stage {}".format(5 - curr_layer)] = (ann_AB, ann_AB_nnd)
        setting.nnf_global["stage {}".format(5 - curr_layer)] = (ann_AB, ann_AB_nnd, ann_BA, ann_BA_nnd)

        if curr_layer >= 4:
            print("### current stage: %d - end | "%(5-curr_layer)+"Elapse: "+str(datetime.timedelta(seconds=time.time()- start_time_1))[:-7]+' ###')
            break

        # The code below is used to initialize the F_AP&F_B in the next layer,
        # it generates the R_B' and R_A as is stated in Equotion(4)
        # R_B' is stored in F_AP, R_A is stored in F_B

        # using backpropagation to approximate feature

        # About why we add 2 here:
        # https://github.com/msracver/Deep-Image-Analogy/issues/30
        next_layer = curr_layer + 2

        ann_AB_upnnf2 = upSample_nnf(ann_AB, F_A_size[next_layer][2:])
        ann_BA_upnnf2 = upSample_nnf(ann_BA, F_B_size[next_layer][2:])

        F_AP_np = avg_vote(ann_AB_upnnf2, ts2np(F_BP[next_layer]), nnf_patch_size[next_layer], F_A_size[next_layer][2:],
                              F_B_size[next_layer][2:])
        F_B_np = avg_vote(ann_BA_upnnf2, ts2np(F_A[next_layer]), nnf_patch_size[next_layer], F_B_size[next_layer][2:],
                             F_A_size[next_layer][2:])

        # Initialize  R_B' and R_A
        F_AP[next_layer] = np2ts(F_AP_np, device)
        F_B[next_layer] = np2ts(F_B_np, device)

        # Warp F_BP using ann_AB, Warp F_A using ann_BA
        target_BP_np = avg_vote(ann_AB, ts2np(F_BP[curr_layer]), nnf_patch_size[curr_layer], F_A_size[curr_layer][2:],
                                F_B_size[curr_layer][2:])
        target_A_np = avg_vote(ann_BA, ts2np(F_A[curr_layer]), nnf_patch_size[curr_layer], F_B_size[curr_layer][2:],
                               F_A_size[curr_layer][2:])

        target_BP = np2ts(target_BP_np, device)
        target_A = np2ts(target_A_np, device)

        # ADAM algorithm to approximate R_B' and R_A
        print('- deconvolution for feat A\'')
        start_time_2 = time.time()
        F_AP[curr_layer+1] = model.get_deconvoluted_feat(target_BP, curr_layer, F_AP[next_layer], lr=lr[curr_layer],
                                                              iters=400, display=False)
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])

        print('- deconvolution for feat B')
        start_time_2 = time.time()
        F_B[curr_layer+1] = model.get_deconvoluted_feat(target_A, curr_layer, F_B[next_layer], lr=lr[curr_layer],
                                                             iters=400, display=False)
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])

        # in case of data type inconsistency
        if F_B[curr_layer + 1].type() == torch.cuda.DoubleTensor:
            F_B[curr_layer + 1] = F_B[curr_layer + 1].type(torch.cuda.FloatTensor)
            F_AP[curr_layer + 1] = F_AP[curr_layer + 1].type(torch.cuda.FloatTensor)

        print("### current stage: %d - end | "%(5-curr_layer)+"Elapse: "+str(datetime.timedelta(seconds=time.time()- start_time_1))[:-7]+' ###')

    print('\n- reconstruct images A\' and B')
    img_AP = reconstruct_avg(ann_AB, img_BP, nnf_patch_size[curr_layer], F_A_size[curr_layer][2:], F_B_size[curr_layer][2:])

    img_AP = np.clip(img_AP, 0, 255)

    return img_AP, time.time()-start_time_0