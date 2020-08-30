def init():
    global optLosses_AP
    global optLosses_B
    optLosses_AP = {"2":[], "3":[], "4":[], "5":[]}
    optLosses_B = {"2": [], "3": [], "4": [], "5": []}

    global isAP
    isAP = True

    global details
    details = ""

    global nnf_global
    nnf_global = {}

    global is_first_frame
    is_first_frame = True

    global is_prev_frame_changed
    global prev_frame
    global currlayer_global

    global is_debug_mode