from init_gui import getparams
"""
init_gui라는 모듈에서 getparams 함수를 가져온다.
"""

from scipy.spatial.transform import Rotation as R
"""
이 코드는 Python의 scipy 라이브러리에서 spatial.transform 모듈 내의 
Rotation 클래스를 가져와 사용하도록 하는 코드입니다. 가져올 때, 
간결하게 사용하기 위해 R이라는 별칭으로 가져오고 있습니다. 

scipy.spatial.transform 모듈은 각종 회전 및 변환 연산(3차원 공간에서의 회전, 축 변환 등)을 
처리하기 위한 도구를 제공합니다. Rotation 클래스는 여기에 속하는 주요 클래스로서,
주로 3차원 공간에서 회전을 표현하고 관리하는데 사용됩니다. 이 클래스를 이용하면 예를 들어,
오일러 각(Euler angles), 이 회전에 대한 축 및 각, 쿼터니언(Quaternion) 등 다양한 표현들을 서로 변환하거나,
회전 연산을 적용할 수 있습니다.
"""

import cv2
"""
cv2 (OpenCV):

cv2는 영상 처리와 컴퓨터 비전을 위한 라이브러리인 OpenCV(Open Source Computer Vision Library)의 파이썬 인터페이스입니다.
영상 데이터를 읽고 쓰기, 영상 처리 및 분석, 객체 검출 및 추적, 카메라 캡처 등 다양한 컴퓨터 비전 작업을 수행할 수 있습니다.
주요 함수로는 영상 파일 읽기/쓰기를 위한 imread(), imwrite(), 이미지 변환을 위한 cvtColor(), 이미지 표시를 위한 imshow() 등이 있습니다.
"""
import json
"""
이 코드는 Python에서 json 모듈을 임포트하는 코드입니다. json 모듈은 Python에서 JSON(JavaScript Object Notation) 형식의 데이터를 처리하는 데 사용되는 도구를 제공합니다. JSON 데이터는 간단한 문자열 형태로 정보를 저장하고 교환하는 데 널리 사용되는 형식입니다. json 모듈의 주요한 기능은 다음과 같습니다:
json.dump(): Python 객체(예: 사전, 리스트, 튜플 등)를 JSON 형식 문자열 또는 파일로 변환 (직렬화)
json.dumps(): Python 객체를 JSON 형식 문자열로 변환
json.load(): JSON 형식의 데이터를 Python 객체로 읽어오기 (역직렬화)
json.loads(): JSON 형식의 문자열을 Python 객체로 변환
이 모듈을 사용하면 JSON 형식의 데이터를 쉽게 읽고 쓸 수 있으며, 웹 API나 파일과 같은 외부 리소스와 데이터를 주고받을 때 유용하게 쓰일 수 있습니다.
"""

"""
class Parameters(): 라는 코드는 Parameters라는 이름의 새로운 클래스를 정의하고 있습니다. 
이 클래스는 사용자로부터 입력받은 파라미터를 관리하기 위한 목적으로 사용되며, 클래스의 인스턴스 변수를 통해 
사용자 입력 값을 저장하고 있습니다. 이를 통해 다른 코드에서 Parameters 클래스의 인스턴스를 이용하여 
사용자 입력 값을 손쉽게 가져오고 사용할 수 있습니다.
"""
class Parameters():
    def __init__(self) -> None:
        param = None
        # `param`이라는 로컬 변수를 선언하고 None으로 초기화합니다. 이 변수는 임시 저장소 역할을 수행합니다.
        while param == None:
       # 사용자로부터 올바른 입력을 받을 때까지 반복문을 실행합니다. `getparams()` 함수를 통해 입력을 받고 
       # `param`이 None이 아닌 값으로 변경되면 반복문을 종료합니다.
            param = getparams()
            # `getparams()` 함수를 호출하여 사용자로부터 입력받은 값을 `param`에 저장합니다. 
            # 이 함수는 사용자로부터 입력받은 값을 반환하는 역할을 수행합니다.

            """
        이 코드의 목적은 사용자로부터 입력을 반드시 받고, 그 값을 처리하여 Parameters 인스턴스에 저장하기 위함입니다. 
        param 변수는 사용자로부터 입력을 얻기 전까지 None 값을 갖고 있으며, 올바른 입력을 받았을 때 반복문이 종료하고 
        param의 값을 인스턴스 변수로 저장하는 과정이 진행됩니다.
            """     

        self.advanced = param["advanced"]
        # 사용자가 입력한 'advanced' 값을 인스턴스 변수 'self.advanced'에 저장합니다. 
        # 이 값은 고급 모드의 활성화 여부를 나타냅니다.
        self.model = param["model_complexity"]
        # 사용자가 입력한 'model_complexity' 값을 인스턴스 변수 'self.model'에 저장합니다.
        # 이 값은 모델의 복잡성을 나타내는 값입니다.
        self.smooth_landmarks = param["smooth_landmarks"]
        # 사용자가 입력한 'smooth_landmarks' 값을 인스턴스 변수 'self.smooth_landmarks'에 저장합니다. 
        # 이 값은 랜드마크의 부드러운 연산 처리 여부를 나타냅니다.
        self.min_tracking_confidence = param["min_tracking_confidence"]
        # 사용자가 입력한 'min_tracking_confidence' 값을 인스턴스 변수 'self.min_tracking_confidence'에 저장합니다. 
        # 이 값은 모델이 호출하는 추적 알고리즘의 최소 신뢰도를 나타냅니다.
        self.static_image = param["static_image"]
        # 사용자가 입력한 'static_image' 값을 인스턴스 변수 'self.static_image'에 저장합니다. 
        # 이 값은 사용할 이미지가 정적인지 여부를 나타냅니다.


        #PARAMETERS:
        self.maximgsize = param["imgsize"]               
        # 사용자가 입력한 'imgsize' 값을 인스턴스 변수 'self.maximgsize'에 저장합니다. 이 값은 이미지의 최대 크기를 제한하기 위한 목적이며,
        # 한 축이 이 값보다 큰 이미지는 축소됩니다. 이를 통해 과도한 메모리 사용을 방지하고 빠른 처리를 유지할 수 있습니다.
        #to prevent working with huge images, images that have one axis larger 
        # than this value will be downscaled.
        self.cameraid = param["camid"]
        # 사용자가 입력한 'camid' 값을 인스턴스 변수 'self.cameraid'에 저장합니다. 
        # 이 값은 USB 웹캠 또는 가상 웹캠을 사용할 때 특정 카메라를 선택하는데 사용됩니다. 
        # 0이 작동하지 않거나 잘못된 카메라를 여는 경우 1-5 등의 다른 숫자를 시도해 볼 수 있습니다.                   
        #to use with an usb or virtual webcam. If 0 doesnt work/opens wrong camera, try numbers 1-5 or so
        #cameraid = "http://192.168.1.102:8080/video"   
        # #to use ip webcam, uncomment this line and change to your ip


        self.hmd_to_neck_offset = [0,-0.2,0.1]    
        #offset of your hmd to the base of your neck, to ensure the tracking is stable even if you look around. 
        # Default is 20cm down, 10cm back.
        # HMD(Head Mounted Display)와 목의 기본 위치 간의 오프셋을 설정합니다. 기본값은 20cm 아래, 10cm 뒤로 설정되어 
        # 있습니다. 이렇게 하면 사용자가 주위를 둘러볼 때 추적이 안정적으로 유지됩니다.
        self.preview_skeleton = param["prevskel"]            
        #if True, whole skeleton will appear in vr 2 meters in front of you. 
        # Good to visualize if everything is working
        # 사용자가 입력한 'prevskel' 값을 인스턴스 변수 'self.preview_skeleton'에 저장합니다. 
        # 이 값이 True로 설정되면, 전체 스켈레톤이 사용자 앞에 있는 가상 환경에서 확인할 수 있습니다. 
        # 이를 통해 모든 것이 정상 작동하는지 직관적으로 확인할 수 있습니다.
        self.dont_wait_hmd = param["waithmd"]                  
        #dont wait for movement from hmd, start inference immediately.
        # 사용자가 입력한 'waithmd' 값을 인스턴스 변수 'self.dont_wait_hmd'에 저장합니다. 
        # 이 값이 True로 설정되면 HMD의 움직임을 기다리지 않고 즉시 추론을 시작합니다.
        self.rotate_image = 0
        # cv2.ROTATE_90_CLOCKWISE # cv2.ROTATE_90_COUTERCLOCKWISE # cv2.ROTATE_180 # None # if you want, rotate the camera
        # 카메라 회전을 위한 변수입니다. 필요한 경우 값을 변경하여 회전시킬 수 있습니다. 예를 들어, cv2.ROTATE_90_CLOCKWISE는 시계 방향으로 90도 회전을 의미합니다.
        self.camera_latency = 0.0
        self.smoothing_1 = 0.0
        self.additional_smoothing_1 = 0.7
        self.smoothing_2 = 0.5
        self.additional_smoothing_2 = 0.9
        #여러 종류의 스무딩과 카메라 지연 시간 변수를 초기화합니다. 이 변수들은 추적 과정에서 흔들림을 최소화하고 부드럽게 만드는 데 도움이 됩니다.

        self.feet_rotation = param["feetrot"]
        # 사용자가 입력한 'feetrot' 값을 인스턴스 변수 'self.feet_rotation'에 저장합니다. 이 값은 발 회전을 조절하기 위한 설정값입니다.
        self.use_hands = param["use_hands"]
        # 사용자가 입력한 'use_hands' 값을 인스턴스 변수 'self.use_hands'에 저장합니다. 이 값은 손을 사용하는지 여부를 결정하는 설정값입니다.
        self.ignore_hip = param["ignore_hip"]
        # 사용자가 입력한 'ignore_hip' 값을 인스턴스 변수 'self.ignore_hip'에 저장합니다. 이 값은 엉덩이를 무시할지 여부를 설정하는 설정값입니다.
        
        self.camera_settings = param["camera_settings"]
        # 사용자가 입력한 'camera_settings' 값을 인스턴스 변수 'self.camera_settings'에 저장합니다. 이 값은 추적에 사용할 카메라의 설정 정보를 포함합니다.
        self.camera_width = param["camera_width"]
        # 사용자가 입력한 'camera_width' 값을 인스턴스 변수 'self.camera_width'에 저장합니다. 이 값은 카메라가 캡처하는 이미지의 가로 크기를 설정합니다.
        self.camera_height = param["camera_height"]
        # 사용자가 입력한 'camera_height' 값을 인스턴스 변수 'self.camera_height'에 저장합니다. 
        # 이 값은 카메라가 캡처하는 이미지의 세로 크기를 설정합니다.

        self.backend = param["backend"]
        # 사용자가 입력한 'backend' 값을 인스턴스 변수 'self.backend'에 저장합니다. 이 값은 사용할 백엔드를 결정합니다.
        self.backend_ip = param["backend_ip"]
        # 사용자가 입력한 'backend_ip' 값을 인스턴스 변수 'self.backend_ip'에 저장합니다. 이 값은 백엔드 서버의 IP 주소를 설정합니다.
        self.backend_port = param["backend_port"]
        # 사용자가 입력한 'backend_port' 값을 인스턴스 변수 'self.backend_port'에 저장합니다. 이 값은 백엔드 서버의 포트 번호를 설정합니다.
        self.webui = param["webui"]
        # 사용자가 입력한 'webui' 값을 인스턴스 변수 'self.webui'에 저장합니다. 이 값은 웹UI를 사용할지 설정하는 값입니다.
     

        #이 코드는 Parameters 클래스의 인스턴스 변수를 초기화하는 역할을 합니다. 각 라인의 의미는 다음과 같습니다.
        self.calib_rot = True # 회전 보정을 위한 변수를 True로 초기화합니다. 이 변수는 회전 보정을 사용할지 설정하는 값입니다.
        self.calib_tilt = True # 기울기 보정을 위한 변수를 True로 초기화합니다. 이 변수는 기울기 보정을 사용할지 설정하는 값입니다.
        self.calib_scale = True # 스케일링 보정을 위한 변수를 True로 초기화합니다. 이 변수는 스케일링 보정을 사용할지 설정하는 값입니다.

        self.recalibrate = False # 재교정을 위한 변수를 False로 초기화합니다. 이 변수는 작업 중에 재교정이 필요한지 설정하는 값입니다.
        
        #rotations in degrees!
        # 회전은 각도(도) 단위로 표현됩니다.
        self.euler_rot_y = 180
        self.euler_rot_x = 90
        self.euler_rot_z = 180
        # 오일러 회전을 위한 변수를 초기화합니다. 이 변수들은 각각 Y, X, Z 축에 대한 회전을 설정하는 값입니다.

        self.posescale = 1     
        # 포즈 스케일링을 위한 변수를 초기화합니다. 이 변수는 포즈 크기를 조절하는 데 사용되는 값입니다.        
        self.exit_ready = False
        # 프로그램 종료 준비를 위한 변수를 False로 초기화합니다. 이 변수는 프로그램이 종료되기 전에 사용자에게 준비를 알리는 용도로 사용됩니다.

        """
        이렇게 초기화된 인스턴스 변수들은 프로그램 내에서 사용되어 다양한 처리를 수행하는 데 사용됩니다. 
        이러한 변수들은 프로그램 실행 중에 사용자의 설정에 따라 작업이 움직입니다. 
        작업을 진행하면서 클래스의 다른 메서드에서 참조되고 사용되게 됩니다.
            """


        self.img_rot_dict = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
        # 이미지 회전을 위한 딕셔너리를 초기화합니다. 이 딕셔너리는 회전 값을 연관시키는 역할을 합니다.
        self.img_rot_dict_rev = {None: 0, cv2.ROTATE_90_CLOCKWISE: 1, cv2.ROTATE_180: 2, cv2.ROTATE_90_COUNTERCLOCKWISE: 3}
        # 회전 값을 이미지에 적용하기 위한 역딕셔너리를 초기화합니다.
        self.paused = False
        # 일시 정지 상태를 나타내는 변수를 초기화합니다. 이 변수는 프로그램이 일시 정지된 상태인지 설정하는 값입니다.
        self.flip = False
        # 카메라를 뒤집을지 여부를 나타내는 변수를 초기화합니다. 이 변수는 뒤집기를 적용할지 설정하는 값입니다.
        self.log_frametime = False
        # 프레임 시간 기록 여부를 나타내는 변수를 초기화합니다. 이 변수는 프레임 시간을 기록할지 설정하는 값입니다.
        self.mirror = False
        # 미러링 여부를 나타내는 변수를 초기화합니다. 이 변수는 프로그램 작동 시 이미지를 거울처럼 뒤집어 보여줄지 설정하는 값입니다.
        self.load_params()
        # 'load_params()' 메서드를 호출하여 사용자가 입력한 파라미터 값을 로드합니다.

        self.global_rot_y = R.from_euler('y',self.euler_rot_y,degrees=True)     
        #default rotations, for 0 degrees around y and x
        # 전역 회전 변수 'self.global_rot_y'를 설정합니다. 이 변수는 Y축 회전값을 기반으로 Rotation 객체를 생성합니다.
        self.global_rot_x = R.from_euler('x',self.euler_rot_x-90,degrees=True) 
        # 전역 회전 변수 'self.global_rot_x'를 설정합니다. 이 변수는 X축 회전값을 기반으로 Rotation 객체를 생성합니다.
        self.global_rot_z = R.from_euler('z',self.euler_rot_z-180,degrees=True) 
        # 전역 회전 변수 'self.global_rot_z'를 설정합니다. 이 변수는 Z축 회전값을 기반으로 Rotation 객체를 생성합니다.
        self.smoothing = self.smoothing_1
        self.additional_smoothing = self.additional_smoothing_1
        # 부드러운 처리를 위한 스무딩 값 및 추가 스무딩 값을 초기화합니다.

        #if advanced mode is disabled, always reset smoothing and camera latency to 0
        # 고급 모드가 비활성화된 경우, 스무딩 및 카메라 지연 값을 항상 0으로 리셋합니다.
        if not self.advanced:
            self.smoothing = 0.0
            self.smoothing_1 = 0.0
            self.camera_latency = 0.0
        
    
        """
    이 코드는 Parameters 클래스 내의 다양한 callback 메서드들을 정의하고 있습니다. 
    이러한 메서드들은 슬라이더의 값이 변경될 때 호출되어 회전 및 크기 조절 등의 작업을 수행하게 됩니다.
        """

    def change_recalibrate(self):
        self.recalibrate = True
        # recalibrate 값을 True로 설정하여 재보정을 수행하도록 지시합니다.

    def rot_change_y(self, value):                                  #callback functions. Whenever the value on sliders are changed, they are called
        print(f"INFO: Changed y rotation value to {value}")
        self.euler_rot_y = value
        self.global_rot_y = R.from_euler('y',value,degrees=True)     #and the rotation is updated with the new value.
        # 슬라이더에서 Y축 회전 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.global_rot_y를 업데이트합니다.


    def rot_change_x(self, value):
        print(f"INFO: Changed x rotation value to {value}")
        self.euler_rot_x = value
        self.global_rot_x = R.from_euler('x',value-90,degrees=True) 
        # 슬라이더에서 X축 회전 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.global_rot_x를 업데이트합니다.


    def rot_change_z(self, value):
        print(f"INFO: Changed z rotation value to {value}")
        self.euler_rot_z = value
        self.global_rot_z = R.from_euler('z',value-180,degrees=True) 
         # 슬라이더에서 Z축 회전 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.global_rot_z를 업데이트합니다.


    def change_scale(self, value):
        print(f"INFO: Changed scale value to {value}")
        #posescale = value/50 + 0.5
        self.posescale = value
        # 슬라이더에서 scale 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.posescale를 업데이트합니다.


    def change_img_rot(self, val):
        print(f"INFO: Changed image rotation to {val*90} clockwise")
        self.rotate_image = self.img_rot_dict[val]
        # 슬라이더에서 이미지 회전 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.rotate_image를 업데이트합니다.

    def change_smoothing(self, val, paramid = 0):
        print(f"INFO: Changed smoothing value to {val}")
        self.smoothing = val
        
        if paramid == 1:
            self.smoothing_1 = val
        if paramid == 2:
            self.smoothing_2 = val
        # 사용자가 smoothing 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.smoothing 및 선택한 paramid에 따라 smoothing_1 또는 smoothing_2 값을 업데이트합니다.


    def change_additional_smoothing(self, val, paramid = 0):
        print(f"INFO: Changed additional smoothing value to {val}")
        self.additional_smoothing = val

        if paramid == 1:
            self.additional_smoothing_1 = val
        if paramid == 2:
            self.additional_smoothing_2 = val
        # 사용자가 추가 smoothing 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.additional_smoothing 및 선택한 paramid에 따라 additional_smoothing_1 또는 additional_smoothing_2 값을 업데이트합니다.


    def change_camera_latency(self, val):
        print(f"INFO: Changed camera latency to {val}")
        self.camera_latency = val
        # 사용자가 카메라 지연 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.camera_latency 값을 업데이트합니다.


    def change_neck_offset(self,x,y,z):
        print(f"INFO: Hmd to neck offset changed to: [{x},{y},{z}]")
        self.hmd_to_neck_offset = [x,y,z]
        # 사용자가 목 오프셋 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.hmd_to_neck_offset 값을 업데이트합니다.

    def change_mirror(self, mirror):
        print(f"INFO: Image mirror set to {mirror}")
        self.mirror = mirror
        # 사용자가 미러 플래그 값을 변경하면 호출되는 메서드입니다. 변경된 값을 사용하여 self.mirror 값을 업데이트합니다.

    def ready2exit(self):
        self.exit_ready = True
        # ready2exit 메서드는 프로그램 종료 준비를 위해 인스턴스 변수 exit_ready 값을 True로 설정합니다.


    def save_params(self):
        param = {}
        param["rotate"] = self.img_rot_dict_rev[self.rotate_image] 
        param["smooth1"] = self.smoothing_1
        param["smooth2"] = self.smoothing_2

        param["camlatency"] = self.camera_latency
        param["addsmooth1"] = self.additional_smoothing_1
        param["addsmooth2"] = self.additional_smoothing_2

        #if self.flip:
        param["roty"] = self.euler_rot_y
        param["rotx"] = self.euler_rot_x
        param["rotz"] = self.euler_rot_z
        param["scale"] = self.posescale
        
        param["calibrot"] = self.calib_rot
        param["calibtilt"] = self.calib_tilt
        param["calibscale"] = self.calib_scale
        
        param["flip"] = self.flip
        
        param["hmd_to_neck_offset"] = self.hmd_to_neck_offset
        
        param["mirror"] = self.mirror
        
        #print(param["roty"])
        
        with open("saved_params.json", "w") as f:
            json.dump(param, f)

        """
    이 코드는 Parameters 클래스 내의 두 개의 메서드를 정의합니다. 
    이 메서드들은 프로그램 종료를 준비하고, 인스턴스 변수를 JSON 파일에 저장하는 데 사용됩니다.
    save_params 메서드는 다양한 인스턴스 변수의 값을 딕셔너리에 저장한 다음, 이를 JSON 파일에 저장합니다. 
    이렇게 저장된 파일은 나중에 다시 사용할 수 있습니다. 사용자의 설정에 따라 업데이트된 변수 값을 저장하고, 
    작업 중 프로그램 종료를 준비하기 위해 이러한 메서드들은 프로그램 실행 도중 다른 메서드와 함께 작동합니다. 
    이 기능을 통해 사용자는 원하는 설정을 저장하고 유지할 수 있습니다.
        """

        """
    이 코드는 Parameters 클래스의 load_params 메서드와 아래에 있는 스크립트의 시작점인 
    if __name__ == "__main__" 부분을 정의합니다. load_params 메서드는 다음과 같습니다.
        """

    def load_params(self):

        try:
            with open("saved_params.json", "r") as f:
                param = json.load(f)

            #print(param["roty"])

            self.rotate_image = self.img_rot_dict[param["rotate"]]
            self.smoothing_1 = param["smooth1"]
            self.smoothing_2 = param["smooth2"]
            self.camera_latency = param["camlatency"]
            self.additional_smoothing_1 = param["addsmooth1"]
            self.additional_smoothing_2 = param["addsmooth2"]

            self.euler_rot_y = param["roty"]
            self.euler_rot_x = param["rotx"]
            self.euler_rot_z = param["rotz"]
            self.posescale = param["scale"]
            
            self.calib_rot = param["calibrot"]
            self.calib_tilt = param["calibtilt"]
            self.calib_scale = param["calibscale"]
            
            self.mirror = param["mirror"]
            
            if self.advanced:
                self.hmd_to_neck_offset = param["hmd_to_neck_offset"]
            
            self.flip = param["flip"]
        except:
            print("INFO: Save file not found, will be created after you exit the program.")
 # load_params 메서드는 JSON 파일에서 인스턴스 변수의 값을 읽고 불러옵니다.
            """
        이 메서드는 이전에 저장된 JSON 파일을 열고, 그 내용을 인스턴스 변수에 로드하여 프로그램에서 사용할 수 있게 합니다. 
        그러나 JSON 파일이 없으면 메서드는 "Save file not found" 메시지를 출력하고, 
        프로그램을 종료할 때 파일을 생성합니다. __name__ == "__main__" 블록은 다음과 같습니다.
           """

if __name__ == "__main__":
    print("hehe")
    # 이 구문은 스크립트의 시작점입니다. 스크립트가 직접 실행될 때만 "hehe"라는 문자열을 출력합니다.

    """
        이 블록은 스크립트가 직접 실행되는 경우에만 print("hehe")를 호출합니다. 
        이것은 주로 스크립트를 개발하거나 디버깅할 때 유용한 상호작용입니다. 
        프로그램의 시작점에서 원하는 작업을 추가할 수 있습니다.
             """