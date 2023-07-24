import tkinter as tk
from tkinter import ttk

import sys
import pickle

import os
import pathlib
import pkg_resources

from sys import platform

def set_advanced(window,param):
    param["switch_advanced"] = True
    window.quit()

def getparams():
    
    try:
        param = pickle.load(open("params.p","rb"))
    except:
        param = {}
        
    if "camid" not in param:
        param["camid"] = 'http://192.168.1.103:8080/video'
    if "imgsize" not in param:
        param["imgsize"] = 640
    if "neckoffset" not in param:
        param["neckoffset"] = [0.0, -0.2, 0.1]
    if "prevskel" not in param:
        param["prevskel"] = False
    if "waithmd" not in param:
        param["waithmd"] = False
    if "rotateclock" not in param:
        param["rotateclock"] = False
    if "rotatecclock" not in param: 
        param["rotatecclock"] = False
    if "rotate" not in param:
        param["rotate"] = None
    if "camlatency" not in param:
        param["camlatency"] = 0.05
    if "smooth" not in param:
        param["smooth"] = 0.5
    if "feetrot" not in param:
        param["feetrot"] = False
    if "calib_scale" not in param:
        param["calib_scale"] = True
    if "calib_tilt" not in param:
        param["calib_tilt"] = True
    if "calib_rot" not in param:
        param["calib_rot"] = True
    if "use_hands" not in param:
        param["use_hands"] = False
    if "ignore_hip" not in param:
        param["ignore_hip"] = False
    if "camera_settings" not in param:
        param["camera_settings"] = False
    if "camera_width" not in param:
        param["camera_width"] = 640
    if "camera_height" not in param:
        param["camera_height"] = 480
    if "model_complexity" not in param:
        param["model_complexity"] = 1
    if "smooth_landmarks" not in param:
        param["smooth_landmarks"] = True
    if "min_tracking_confidence" not in param:
        param["min_tracking_confidence"] = 0.5
    if "static_image" not in param:
        param["static_image"] = False    
    if "backend" not in param:
        param["backend"] = 1
    if "backend_ip" not in param:
        param["backend_ip"] = "127.0.0.1"
    if "backend_port" not in param:
        param["backend_port"] = 9000
    if "advanced" not in param:
        param["advanced"] = False
    if "webui" not in param:
        param["webui"] = False

        
    window = tk.Tk()
    window.configure(bg='#D3F0F2')
    
    window.title("Logy 1.1v Dev(hong)")

    image_path = pkg_resources.resource_filename(__name__, 'templates/logy.png')
    image = tk.PhotoImage(file=image_path)
    
    label=tk.Label(window, image=image,bg='#D3F0F2')     
    label.pack()

    def on_close():
        window.destroy()
        sys.exit("INFO: Exiting... You can close the window after 10 seconds.")

    window.protocol("WM_DELETE_WINDOW", on_close)

    if not param["advanced"]:
        tk.Label(text="The format http://<ip-here>:8080/video is for use with\n the IP Webcam android application.\n If using something else, such as a regular USB Webcam,\n just try 0,1,2... until the correct camera opens.", width = 50,bg='#D3F0F2').pack()

    tk.Label(text="Camera IP or ID:", width = 50, bg = "gray19", fg = 'gray60').pack()
    camid = tk.Entry(width = 50)
    camid.pack()
    camid.insert(0,param["camid"])
    
    if not param["advanced"]:
        tk.Label(text="NOTE: Increasing resolution may decrease performance.\n Unless you have problems with opening camera, \nleave it as default.", width = 50,bg='#D3F0F2').pack()


    ######################################
    # 레이블과 입력 상자를 포함할 프레임 생성
    frame = tk.Frame(window, bg='#D3F0F2')
    frame.pack()

    # 텍스트 레이블 생성 및 프레임에 추가
    label = tk.Label(frame, text="Camera width:", bg='#C8FAC6')
    label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

    # 입력 상자 생성 및 프레임에 추가
    camwidth = tk.Entry(frame, width=20)
    camwidth.grid(row=0, column=1, sticky="w", padx=5, pady=5)
    camwidth.insert(0, param["camera_width"])

    # 'Camera height' 텍스트 레이블 생성 및 프레임에 추가
    label_height = tk.Label(frame, text="Camera height:", bg='#C8FAC6')
    label_height.grid(row=1, column=0, sticky="w", padx=5, pady=5)

    # 'Camera height' 입력 상자 생성 및 프레임에 추가
    camheight = tk.Entry(frame, width=20)
    camheight.grid(row=1, column=1, sticky="w", padx=5, pady=5)
    camheight.insert(0, param["camera_height"])


    if platform == "win32":
        if not param["advanced"]:
            tk.Label(text="NOTE: Opening camera settings may change camera behaviour. \nSome cameras may only work with this enabled, some only\n with this disabled, and it may change which camera ID\n you have to use.", width = 55, bg='#D3F0F2').pack()
    
    varcamsettings = tk.IntVar(value = param["camera_settings"])
    cam_settings_check = tk.Checkbutton(text = "Attempt to open camera settings", variable = varcamsettings, bg='#D3F0F2')
    if platform == "win32":
        cam_settings_check.pack()

    if param["advanced"]:
        # 'Maximum image size' 텍스트 레이블 생성 및 프레임에 추가
        label_maximgsize = tk.Label(frame, text="Maximum image size:", bg='#C8FAC6')
        label_maximgsize.grid(row=2, column=0, sticky="w", padx=5, pady=5)

        # 'Maximum image size' 입력 상자 생성 및 프레임에 추가
        maximgsize = tk.Entry(frame, width=20)
        maximgsize.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        maximgsize.insert(0, param["imgsize"])

    ###가로선###
    canvas = tk.Canvas(window, width=350, height=1, bg='black', highlightthickness=0)  # Adjust the width and height as needed
    canvas.pack()

    varfeet = tk.IntVar(value = param["feetrot"])
    rot_feet_check = tk.Checkbutton(text = "Enable experimental foot rotation", variable = varfeet, bg='#D3F0F2')
    rot_feet_check.pack()
    
    if not param["advanced"]:
        tk.Label(text="NOTE: VRChat works better with a hip tracker. Only disable \nit if you use another software for hip tracking, such as owoTrack.", width = 55,bg='#D3F0F2').pack()
    
    varhip = tk.IntVar(value = param["ignore_hip"])
    hip_check = tk.Checkbutton(text = "Disable hip trackerㅤ  ㅤㅤㅤㅤㅤ", variable = varhip, bg='#D3F0F2')
    hip_check.pack()
    
    ###가로선###
    canvas = tk.Canvas(window, width=350, height=1, bg='black', highlightthickness=0)  # Adjust the width and height as needed
    canvas.pack()
    
    if param["advanced"]:
        
        varhand = tk.IntVar(value = param["use_hands"])
        hand_check = tk.Checkbutton(text = "DEV: Spawn trackers for hands ㅤ", variable = varhand, bg='#D3F0F2')
        hand_check.pack()
        
        varskel = tk.IntVar(value = param["prevskel"])
        skeleton_check = tk.Checkbutton(text = "DEV: Preview skeleton in VRㅤ  ㅤ", variable = varskel, bg='#D3F0F2')
        skeleton_check.pack()

        tk.Label(text="[ADVANCED] MediaPipe estimator parameters:", width = 50, bg = "gray19", fg = 'gray60').pack()
        
        ######################################
        # 새 프레임 생성
        frame_params = tk.Frame(bg='#D3F0F2')
        frame_params.pack(padx=5, pady=5)

        # Model complexity 라벨
        label_modelc = tk.Label(frame_params, text="Model complexity:", bg='#C8FAC6')
        label_modelc.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Model complexity 입력 상자
        modelc = tk.Entry(frame_params, width=20)
        modelc.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        modelc.insert(0, param["model_complexity"])


        varmsmooth = tk.IntVar(value = param["smooth_landmarks"])
        msmooth_check = tk.Checkbutton(text = "Smooth landmarksㅤㅤㅤㅤㅤㅤ  ", variable = varmsmooth, bg='#D3F0F2')
        msmooth_check.pack()
        
        # Min tracking confidence 라벨
        label_trackc = tk.Label(frame_params, text="Min tracking confidence:", bg='#C8FAC6')
        label_trackc.grid(row=1, column=0, sticky="w", padx=5, pady=5)

        # Min tracking confidence 입력 상자
        trackc = tk.Entry(frame_params, width=20)
        trackc.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        trackc.insert(0, param["min_tracking_confidence"])

        varstatic = tk.IntVar(value = param["static_image"])
        static_check = tk.Checkbutton(text = "Static image modeㅤㅤㅤㅤㅤㅤ  ", variable = varstatic, bg='#D3F0F2')
        static_check.pack()

    backend_frame = tk.Frame(window)
    backend_selection_frame = tk.Frame(backend_frame)
    backend_options_frame = tk.Frame(backend_frame)

    varbackend = tk.IntVar(value = param["backend"])

    def show_hide_backend_options():
        if varbackend.get() == 2:
            backend_options_frame.pack(side = tk.BOTTOM)
        else:
            backend_options_frame.pack_forget()

    tk.Label(backend_selection_frame, text="Backend: ", bg='#D3F0F2').pack(side = tk.LEFT)
    tk.Radiobutton(backend_selection_frame, text="SteamVR", variable = varbackend, value = 1, bg='#D3F0F2',command = show_hide_backend_options).pack(side = tk.LEFT)
    tk.Radiobutton(backend_selection_frame, text="VRChatOSC",  variable = varbackend, value = 2, bg='#D3F0F2', command = show_hide_backend_options).pack(side = tk.LEFT)
    backend_selection_frame.pack(side = tk.TOP)

    tk.Label(backend_options_frame, text="IP/port:").pack(side = tk.LEFT)
    backend_ip = tk.Entry(backend_options_frame, width = 15)
    backend_ip.insert(0, param["backend_ip"])
    backend_ip.pack(side = tk.LEFT)
    backend_port = tk.Entry(backend_options_frame, width = 5)
    backend_port.insert(0, param["backend_port"])
    backend_port.pack(side = tk.LEFT)

    show_hide_backend_options()
    backend_frame.pack()

    #####가로선#####
    canvas = tk.Canvas(window, width=350, height=1, bg='black', highlightthickness=0)  # Adjust the width and height as needed
    canvas.pack()

    varwebui = tk.IntVar(value = param["webui"])
    webui_check = tk.Checkbutton(text = "Enable webui to control parameters from another device", variable = varwebui, bg='#D3F0F2')
    webui_check.pack()
    
    param["switch_advanced"] = False
    if param["advanced"]:
        #####가로선#####
        canvas = tk.Canvas(window, width=350, height=1, bg='black', highlightthickness=0)  # Adjust the width and height as needed
        canvas.pack()
        tk.Button(text='Disable advanced mode', bg='white',command=lambda *args: set_advanced(window,param)).pack()
    else:
        tk.Button(text='Enable advanced mode', bg='white',command=lambda *args: set_advanced(window,param)).pack()

    tk.Button(text=' ㅤSave and Continueㅤ', bg='white',command=window.quit).pack()

    window.mainloop()

#----------------------------------------------------------"

    cameraid = camid.get()
    #hmd_to_neck_offset = [float(val) for val in hmdoffsettext.get().split(" ")]
    
    dont_wait_hmd = False #bool(varhmdwait.get()) 
    
    #camera_latency = float(camlatencytext.get())
    #smoothing = float(smoothingtext.get())
    feet_rotation = bool(varfeet.get())
    
    ignore_hip = bool(varhip.get())
    camera_settings = bool(varcamsettings.get())
    camera_height = camheight.get()
    camera_width = camwidth.get()
    
    backend = int(varbackend.get())
    backend_ip_set = backend_ip.get()
    backend_port_set = int(backend_port.get())
    
    webui = bool(varwebui.get())
    
    if param["advanced"]:
        maximgsize = int(maximgsize.get())
        
        preview_skeleton = bool(varskel.get())
        use_hands = bool(varhand.get())
        
        mp_smoothing = bool(varmsmooth.get())
        model_complexity = int(modelc.get())
        min_tracking_confidence = float(trackc.get())
        static_image = bool(varstatic.get())
        
    else:
        maximgsize = 640
        
        preview_skeleton = False
        use_hands = False
        
        mp_smoothing = True
        model_complexity = 1
        min_tracking_confidence = 0.5
        static_image = False

    switch_advanced = param["switch_advanced"]

    advanced = param["advanced"]

    param = {}
    param["camid"] = cameraid
    param["imgsize"] = maximgsize
    #param["neckoffset"] = hmd_to_neck_offset
    param["prevskel"] = preview_skeleton
    param["waithmd"] = dont_wait_hmd

    #param["smooth"] = smoothing
    #param["camlatency"] = camera_latency
    param["feetrot"] = feet_rotation
    param["use_hands"] = use_hands
    param["ignore_hip"] = ignore_hip
    
    param["camera_settings"] = camera_settings
    param["camera_height"] = camera_height
    param["camera_width"] = camera_width
    
    param["model_complexity"] = model_complexity
    param["smooth_landmarks"] = mp_smoothing
    param["static_image"] = static_image
    param["min_tracking_confidence"] = min_tracking_confidence
    param["backend"] = backend
    param["backend_ip"] = backend_ip_set
    param["backend_port"] = backend_port_set
    param["webui"] = webui
    
    if switch_advanced:
        param["advanced"] = not advanced
    else:
        param["advanced"] = advanced
    
    pickle.dump(param,open("params.p","wb"))
    
    window.destroy()
    
    if switch_advanced:
        return None
    else:
        return param

if __name__ == "__main__":
    print(getparams())
