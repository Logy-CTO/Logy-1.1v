import time
'''
import time: Python의 내장 time 라이브러리를 임포트하여 시간 관련 함수를 사용할 수 있게 합니다. 
예를 들어, time.sleep() 함수를 사용하여 코드 실행을 일시 중지할 수 있습니다.
'''

from abc import ABC, abstractmethod
'''
from abc import ABC, abstractmethod: Python의 내장 abc (Abstract Base Class) 라이브러리에서 
ABC 클래스와 abstractmethod 데코레이터를 임포트합니다. 이를 사용하여 추상 클래스를 정의하고, 
해당 클래스에서 반드시 구현해야 하는 추상 메서드를 지정할 수 있습니다.
'''

from helpers import  sendToSteamVR
'''
from helpers import sendToSteamVR: helpers 모듈에서 sendToSteamVR 함수를 임포트 합니다. 
이는 SteamVR에 데이터를 전송하는데 사용됩니다.
'''

from scipy.spatial.transform import Rotation as R

from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder
from pythonosc import udp_client
'''
from pythonosc import osc_bundle_builder, osc_message_builder, udp_client: 
pythonosc 라이브러리에서 OSC (Open Sound Control) 통신을 처리하는데 필요한 모듈들을 임포트합니다. 
이 임포트들은 애플리케이션 간에 실시간으로 데이터를 교환할 때 사용되며, 
여기서는 OSC 메시지의 생성, 번들링 및 UDP 클라이언트를 통한 전송을 위해 필요한 모듈들을 가져옵니다.
'''

from helpers import shutdown
import numpy as np

'''
이 코드는 추상 클래스 Backend를 정의하고 있습니다. 
추상 클래스는 구체적인 구현이 없는 메서드를 가질 수 있는 클래스입니다. 
이 경우 ABC(Abstract Base Class)를 상속받아 Backend 클래스를 추상 클래스로 만들었습니다. 
Backend 클래스는 다음과 같은 추상 메서드들을 포함하고 있습니다.
'''
class Backend(ABC):

    @abstractmethod
    def onparamchanged(self, params):
        ...
###추상 메서드 onparamchanged는 파라미터가 변경되었을 때 사용됩니다. 하위 클래스에 의해 반드시 구현되어야 하며, 구체적인 구현은 하위 클래스에 달려있습니다.

    @abstractmethod
    def connect(self, params):
        ...
###추상 메서드 connect는 백엔드에 연결하는 데 사용됩니다. 이 메서드 역시 하위 클래스에서 구현되어야 하며, 연결 프로세스는 상황에 따라 다를 수 있습니다.

    @abstractmethod
    def updatepose(self, params, pose3d, rots, hand_rots):
        ...
###추상 메서드 updatepose는 객체의 자세(pose)를 업데이트 하는 데 사용됩니다. 하위 클래스에서 구체적으로 구현되어야 합니다.
# 매개변수 params, pose3d, rots, hand_rots는 자세를 업데이트하는 데 필요한 데이터를 전달하는 데 사용됩니다.

    @abstractmethod
    def disconnect(self):
        ...
###추상 메서드 disconnect는 백엔드와의 연결을 종료하는 데 사용됩니다. 
# 이 메서드는 하위 클래스에서 구체적으로 구현되며, 연결 종료에 따라 리소스를 정리하고 초기화하는 작업을 수행합니다.


'''
이러한 추상 메서드들은 하위 클래스에서 반드시 구현되어야 하며, 하위 클래스는 상황에 따라 이러한 메서드들을 구체화하여 필요한 작업을 수행할 수 있습니다. 
Backend 클래스의 목적은 백엔드와 관련된 작업에 대한 템플릿(또는 인터페이스)을 제공하는 것입니다. 
각각의 하위 클래스는 이 인터페이스에 맞춰 해당 백엔드 구현에 필요한 작업을 수행하게 됩니다.
'''

class DummyBackend(Backend):

    def __init__(self, **kwargs):
        pass

    def onparamchanged(self, params):
        pass

    def connect(self, params):
        pass

    def updatepose(self, params, pose3d, rots, hand_rots):
        pass

    def disconnect(self):
        pass
'''
해당 코드는 Backend 추상 클래스를 상속받아 새로운 클래스 DummyBackend를 정의하고 있습니다. 
DummyBackend 클래스는 하나의 생성자와 추상 클래스 Backend로부터 상속받았던 4개의 추상 메서드를 구현하고 있습니다. 
하지만, 이 클래스는 메서드 내부에서 어떠한 동작도 수행하지 않고, 모든 구현 내용은 pass로 처리하고 있습니다. 
즉, 이 클래스는 실제로 어떤 동작도 하지 않고, 오직 Backend 추상 클래스에 정의된 메서드들을 구현하기 위한 목적으로만 사용됩니다. 
즉, 이 클래스를 상속받아서 각각의 하위 클래스에서 필요한 작업을 구체적으로 수행하는 방식으로 사용되며, 이 과정에서 필요한 추상 메서드들을 구현하게 됩니다. 
이렇게 함으로써 Backend 클래스와 DummyBackend 클래스 사이의 인터페이스를 유지하면서, 실제 백엔드 구현은 각각의 하위 클래스에서 차이를 두며 수행될 수 있습니다.
'''

class SteamVRBackend(Backend):

    def __init__(self, **kwargs):
        pass
    ###. __init__ 메서드는 생성자 함수이며, 특별한 작업은 수행하지 않습니다. 따라서 pass를 사용하였습니다.

    def onparamchanged(self, params):
        resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
        if resp is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)
    ###onparamchanged 메서드는 매개변수 params를 입력받습니다. 
    # 이 메서드는 스팀VR의 **스무딩(smoothing)과 추가적인** 스무딩(additional_smoothing) 값을 설정하는 것을 처리합니다. 
    # 메서드 내부에서는 sendToSteamVR() 함수를 사용하여 스팀VR에 스무딩과 추가적인 스무딩 값을 보내줍니다. 
    # 만약 sendToSteamVR() 함수가 응답하지 않을 경우, "ERROR" 메시지를 출력하고 shutdown() 함수를 호출하여 프로그램을 종료합니다.

    def connect(self, params):
        print("Connecting to SteamVR")
    ###해당 코드는 SteamVRBackend 클래스의 connect 메서드를 정의하고 있습니다. 
    # connect 메서드는 매개변수 params를 입력받습니다. 이 메서드는 스팀VR과의 연결을 설정하는 작업을 수행합니다. 

        ###먼저, "Connecting to SteamVR" 메시지를 출력합니다.

        numtrackers = sendToSteamVR("numtrackers")
        ### 그 후, 스팀VR에 연결된 기기의 수를 확인하여 numtrackers 변수에 저장합니다.
        if numtrackers is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            ###만약 sendToSteamVR 함수가 응답하지 않을 경우, "ERROR" 메시지를 출력하고 shutdown 함수를 호출하여 프로그램을 종료합니다. 
            shutdown(params)

        numtrackers = int(numtrackers[2])

        #games use 3 trackers, but we can also send the entire skeleton if we want to look at how it works

        totaltrackers = 23 if params.preview_skeleton else  3
        ###이어서, totaltrackers 변수에 할당될 트래커의 수를 결정합니다. 
        # params.preview_skeleton이 True라면 totaltrackers 변수를 23으로 할당하여 전신 및 손 모션 등을 미리 예측할 수 있도록 합니다.

        if params.use_hands:
            totaltrackers = 5
            ###만약 params.use_hands가 True이면, totaltrackers 변수를 5로 할당하여 손 모션만 예측하도록 합니다. 
        if params.ignore_hip:
            totaltrackers -= 1
            ###params.ignore_hip가 True이면, 무효로 처리되는 1개의 트래커가 트래커 수에서 제외됩니다. 

        roles = ["TrackerRole_Waist", "TrackerRole_RightFoot", "TrackerRole_LeftFoot"]
        ###roles 변수에는 트래커가 SteamVR에서 사용되는 역할(role) 문자열을 담은 리스트를 할당합니다. 따라서, 설정된 트래커 수 만큼 트래커 역할(role)이 설정됩니다.
        
        if params.ignore_hip and not params.preview_skeleton:
            del roles[0]
        ###params.ignore_hip가 True이고, params.preview_skeleton가 False인 경우, roles에 저장되어 있던 첫 번째 값을 삭제합니다. 이 경우, 힙(Hip)또는 엉덩이(Buttock) 위치를 추적하지 않습니다.

        if params.use_hands:
            roles.append("TrackerRole_Handed")
            roles.append("TrackerRole_Handed")
        ###params.use_hands가 True이면, roles 리스트에 TrackerRole_Handed를 두 번 추가합니다. 

        for i in range(len(roles),totaltrackers):
            roles.append("None")
        ###그 후, roles 리스트의 길이가 totaltrackers와 동일하게 되도록 남은 리스트를 None 값으로 채웁니다.

        for i in range(numtrackers,totaltrackers):
            #sending addtracker to our driver will... add a tracker. to our driver.
            resp = sendToSteamVR(f"addtracker MPTracker{i} {roles[i]}")
            if resp is None:
                print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
                shutdown(params)
        ###numtrackers에서 totaltrackers까지 반복문을 실행하면서, 
        # 호출할 트래커 번호를 나타내는 i와 그 역할(role)을 설정하기 위한 roles[i]를 파라미터로 전달하여 sendToSteamVR 함수를 호출하여 트래커를 추가합니다.

        resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
        if resp is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)
        ###이를 통해 적절한 트래커 번호와 위치, 역할을 같게 설정할 수 있습니다. 
        # 위 코드의 마지막 부분에서는 다시 한번 스무딩 값을 settings 명령어를 이용하여 스팀 VR에 전달합니다. 
        # 코드는 이전과 같이, 응답이 없으면 프로그램을 종료합니다.





    '''
해당 코드는 SteamVRBackend 클래스의 updatepose 메서드를 구현하는 코드입니다. 
이 메서드는 인자로 params, pose3d, rots, hand_rots를 받습니다. 
params는 이전에 설정된 파라미터 값들(예: preview_skeleton, use_hands 등)을 객체로 가지고 있으며, 
***pose3d, rots, hand_rots***는 최신의 스켈레톤 정보(position과 rotation 정보)를 가지고 있는 3차원 배열입니다. 
    '''

    def updatepose(self, params, pose3d, rots, hand_rots):
        array = sendToSteamVR("getdevicepose 0")        #get hmd data to allign our skeleton to
    ###먼저, sendToSteamVR 함수를 통해 hmd(Head Mounted Display)와 관련된 데이터를 getdevicepose 명령어를 사용하여 가져옵니다. 
    # 스팀VR에 연결되어 있는 HMD를 사용하여, 0번 디바이스 (즉, HMD)의 스켈레톤 정보를 가져옵니다. 

        if array is None or len(array) < 10:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)
            ###array가 None이거나 길이가 10보다 작은 경우, 스팀VR과의 연결이 끊어졌다는 에러 메시지를 출력합니다. 
            # 이 경우 shutdown 메서드를 호출하여 프로그램을 종료합니다. 
            # 위에서 언급한 array는 가져온 0번 디바이스의 스켈레톤 정보입니다. 즉, array는 디바이스의 위치, 회전 quaternion 등을 저장하고 있습니다.


        headsetpos = [float(array[3]),float(array[4]),float(array[5])]
        ###headsetpos는 hmd의 위치 정보를 가지고 있고, [float(array[3]),float(array[4]),float(array[5])]을 통해 수집할 수 있습니다. array의 4, 5, 6번째 인덱스에 위치 정보가 저장되어 있습니다.

        headsetrot = R.from_quat([float(array[7]),float(array[8]),float(array[9]),float(array[6])])
        ###headsetrot에 대한 R 객체(Rodrigues rotation object)를 생성합니다. 
        # 이 Rodrigues rotation object는 array의 6~9번째 값을 적용하여 생성됩니다. 이것은 HMD의 회전 값입니다. 
        # 따라서, headsetpos 및 headsetrot 정보를 사용하여 Neck Joint의 위치를 정확하게 계산할 수 있습니다.


        neckoffset = headsetrot.apply(params.hmd_to_neck_offset)   #the neck position seems to be the best point to allign to, as its well defined on
                                                            #the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.
        ###rots와 pose3d 배열도 새로운 값을 할당 받습니다. 다음으로는 neckoffset를 계산합니다. 
        # 
        # 이것은 HMD/Hip의 위치와 Neck Joint 사이의 거리를 기준으로 Neck Joint의 위치를 조절합니다. 
        # 이를 통해 머리의 운동은 더 잘 추적될 수 있습니다.

        if params.recalibrate:
            print("INFO: frame to recalibrate")
        ###params.recalibrate가 True이면, 재보정 모드입니다. 따라서 프로그램은 이전에 보정된 스켈레톤 정보를 사용하지 않고, 새로운 스켈레톤 정보를 입력받습니다.

        else:
            pose3d = pose3d * params.posescale     #rescale skeleton to calibrated height
            ###pose3d 배열을 params.posescale 값을 사용하여 rescale 해줍니다. 
            #print(pose3d)

            offset = pose3d[7] - (headsetpos+neckoffset)    #calculate the position of the skeleton
            ###이후, headsetpos와 neckoffset을 사용하여 다리, 팔, 머리 등의 관절 위치를 계산합니다.
            ###스켈레톤 좌표를 (8번째 인덱스인 head 관절 위치 - 머리부분, headsetpos - 헤드셋의 중심, neckoffset - 목의 위치)를 이용하여 보정합니다. 
            # 이를 통해 다리, 팔, 머리 등의 위치에 대한 세부 정보를 계산할 수 있습니다.

            if not params.preview_skeleton:
                ###params.preview_skeleton 값이 FALSE인 경우, 실제 스켈레톤을 작업에 적용합니다.
                numadded = 3
                
                if not params.ignore_hip:
                    for i in [(0,1),(5,2),(6,0)]:
                        joint = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {params.camera_latency} 0.8")
               
                else:
                    for i in [(0,1),(5,2)]:
                        joint = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(f"updatepose {i[1]} {joint[0]} {joint[1]} {joint[2]} {rots[i[1]][3]} {rots[i[1]][0]} {rots[i[1]][1]} {rots[i[1]][2]} {params.camera_latency} 0.8")
                        numadded = 2
               
                if params.use_hands:
                    for i in [(10,0),(15,1)]:
                        joint = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        sendToSteamVR(f"updatepose {i[1]+numadded} {joint[0]} {joint[1]} {joint[2]} {hand_rots[i[1]][3]} {hand_rots[i[1]][0]} {hand_rots[i[1]][1]} {hand_rots[i[1]][2]} {params.camera_latency} 0.8")
                '''
            위 코드는 각 관절 별 joint 값을 정의하고, sendToSteamVR 함수를 사용하여 해당 값을 SteamVR에 전송합니다.
            params.ignore_hip 플래그가 True이면은 엉덩이 관절을 처리하지 않습니다. 
            이러한 인덱스 쌍이 모두 결정되면, it for 문은 각 관절 위치를 joint 값에 대입하고 sendToSteamVR 함수를 전송합니다. 
            또한, numadded 변수는 후속 관절 처리용으로 사용됩니다.
               '''

            else:
                for i in range(23):
                    joint = pose3d[i] - offset      #if previewing skeleton, send the position of each keypoint to steamvr without rotation
                    sendToSteamVR(f"updatepose {i} {joint[0]} {joint[1]} {joint[2] - 2} 1 0 0 0 {params.camera_latency} 0.8")
                    ###마지막으로, params.preview_skeleton 플래그가 True이면, 모든 키 포인트를 SteamVR에 전송합니다. 이를 통해 실제 스켈레톤에 대한 프리뷰 작업을 수행할 수 있습니다.

        return True


    def disconnect(self):
        pass

def osc_build_msg(name, position_or_rotation, args):
    builder = osc_message_builder.OscMessageBuilder(address=f"/tracking/trackers/{name}/{position_or_rotation}")
    builder.add_arg(float(args[0]))
    builder.add_arg(float(args[1]))
    builder.add_arg(float(args[2]))
    return builder.build()
'''
이 함수는 OSC 메시지를 작성하는데 사용됩니다. 
위치(Position) 또는 회전(Orientation)을 나타내는 position_or_rotation과 Vive Tracker의 이름으로 구성된 메시지 주소를 만들고,
 Vive Tracker의 위치 또는 회전 값을 메시지에 추가합니다. 최종 메시지는 builder.build()에 의해 생성됩니다. 
 예를 들어 name이 "Tracker001", position_or_rotation이 "position"이고 args가 [-30, 40, 50]인 경우, 
 이 함수는 OSC 메시지를 다음과 같이 빌드합니다.

 /tracking/trackers/Tracker001/position -30.0 40.0 50.0

즉, 메시지 주소는 /tracking/trackers/Tracker001/position이 되며, 값은 -30.0, 40.0, 50.0이 됩니다. 
완성된 OSC 메시지는 VRChatOSCBackend 클래스의 osc_build_bundle 함수에서 builder.add_content(osc_build_msg(...))를 통해 
전체 Vive Tracker 데이터 묶음에 추가됩니다.
'''

def osc_build_bundle(trackers):
    builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    builder.add_content(osc_build_msg(trackers[0]['name'], "position", trackers[0]['position']))
    for tracker in trackers[1:]:
        builder.add_content(osc_build_msg(tracker['name'], "position", tracker['position']))
        builder.add_content(osc_build_msg(tracker['name'], "rotation", tracker['rotation']))
    return builder.build()

'''
이 함수는 모든 Vive Tracker의 위치(Position) 및 회전(Orientation) 값을 OSC 메시지로 묶어서 Unity에서 수신할 수 있도록 만듭니다. 
먼저 osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)로 OSC 번들 빌더를 초기화합니다.
IMMEDIATELY mode로 번들을 빌드하면 번들의 모든 메시지가 동시에 보내지게 됩니다. 
그런 다음, 첫 번째 루프에서 Vive Tracker 데이터의 첫 번째 항목의 위치(Position) 값을 메시지에 추가합니다.
이 위치 데이터는 OSC 메시지 주소가 "position"인 /tracking/trackers/<Vive Tracker Name>/position에 추가됩니다. 
그 후 두 번째 루프에서 첫 번째 Vive Tracker 데이터 이후의 나머지 Vive Tracker들에 대해 position 및 rotation 값을 OSC 메시지로 추가합니다. 
이 데이터는 각각 OSC 주소가 /tracking/trackers/<Vive Tracker Name>/position 및 /tracking/trackers/<Vive Tracker Name>/rotation 
인 메시지에 추가됩니다. 그리고 최종적으로 모든 메시지를 번들 빌더에 추가한 후 builder.build()로 VRChat에 전달할 단일 OSC 번들을 반환합니다.
예를 들어 다음과 같은 Vive Tracker 데이터가 있다고 가정해봅시다:

[
  {
    "name": "Tracker001",
    "position": [-30, 40, 50],
    "rotation": [0.0, 0.0, 0.0, 1.0]
  },
  {
    "name": "Tracker002",
    "position": [10, 20, 30],
    "rotation": [0.0, 0.7071, 0.0, 0.7071]
  },
  {
    "name": "Tracker003",
    "position": [-60, 80, 90],
    "rotation": [0.0, 0.0, -0.7071, 0.7071]
  }
]


이 경우 osc_build_bundle 함수는 다음 OSC 번들을 생성합니다

# OSC Bundle

- OSC Message:
    Address: /tracking/trackers/Tracker001/position
    Arguments: -30.0, 40.0, 50.0
- OSC Message:
    Address: /tracking/trackers/Tracker002/position
    Arguments: 10.0, 20.0, 30.0
- OSC Message:
    Address: /tracking/trackers/Tracker002/rotation
    Arguments: 0.0, 0.7071, 0.0, 0.7071
- OSC Message:
    Address: /tracking/trackers/Tracker003/position
    Arguments: -60.0, 80.0, 90.0
- OSC Message:
    Address: /tracking/trackers/Tracker003/rotation
    Arguments: 0.0, 0.0, -0.7071, 0.7071


결과적으로 Unity에서는 이러한 OSC 번들을 받아서 사용자의 플레이어가 Vive Tracker 데이터를 실시간으로 추적할 수 있습니다.
'''

class VRChatOSCBackend(Backend):
###VRChatOSCBackend 클래스는 Backend 클래스를 상속받아, VRChat에 Vive Tracker 데이터를 전송하는 데 필요한 OSC 통신을 처리하는 백엔드 기능을 담당합니다. 

    def __init__(self, **kwargs):
        self.prev_pose3d = np.zeros((29,3))
        pass
    ###__init__ 함수에서는 이전 프레임에서의 Vive Tracker 위치(Position) 값 (3차원 배열)을 저장하는 prev_pose3d 변수를 0으로 초기화합니다.

    def onparamchanged(self, params):
        pass ###onparamchanged 함수는 이 클래스에서 사용되지 않으므로 빈 함수로 남겨져 있습니다.

    def connect(self, params):
        if hasattr(params, "backend_ip") and hasattr(params, "backend_port"):
            ###connect 함수에서는 params 에 backend_ip과 backend_port가 있는지 검사하고 있다가,
            self.client = udp_client.UDPClient(params.backend_ip, params.backend_port)
            ###connect 함수는 파라미터로 params를 받으며, 이 파라미터는 "backend_ip" 및 "backend_port"와 같은 
            # VRChat에 연결할 백엔드 서버 정보를 담고 있습니다. 즉, 이 클래스를 사용하기 위해서는 VRChat과 백엔드 서버가 먼저 설정되어 있어야 하며, 
            # params 객체를 통해 이를 전달해주어야 합니다.

        else:
            self.client = udp_client.UDPClient("127.0.0.1", 9000)
            ###없으면 기본값인 ("127.0.0.1", 9000)으로 초기화하여 UDP 클라이언트를 생성합니다. 그리고 생성한 클라이언트 객체를 self.client 변수에 저장합니다.
    

    def updatepose(self, params, pose3d, rots, hand_rots):
    ###updatepose 함수는 params, pose3d, rots, hand_rots 파라미터를 받습니다.

        #pose3d[:,1] = -pose3d[:,1]      #flip the positions as coordinate system is different from steamvr
        #pose3d[:,0] = -pose3d[:,0]
        
        pose3d = self.prev_pose3d*params.additional_smoothing + pose3d*(1-params.additional_smoothing)
        self.prev_pose3d = pose3d
        ###self.prev_pose3d는 이전 프레임에서의 Vive Tracker 위치(Position) 값(3차원 배열)입니다. 
        # 이전 프레임에서의 값과 현재 프레임에서의 값을 미리 정해둔 가중치 비율에 따라 smoothing한 값을 사용합니다. 
        # pose3d 값은 이전 프레임에서의 값을 가중치 params.additional_smoothing로 곱해 더한 값과 현재 프레임에서의 값을 
        # 가중치 (1-params.additional_smoothing)으로 곱한 값을 더하여 생성됩니다.


        '''
이 식은 위치 데이터를 부드럽게 보간하는 데 사용된 가중치를 적용하는 방법입니다. 이를 통해 전체적인 프레임의 부드러운 이동을 제공합니다. 이렇게 부드러운 보간을 적용하는 이유는 보통 입력 데이터가 노이즈가 포함되어 있을 수 있고 또는 선택한 프레임 레이트로 인해 데이터 간 시간 간격이 크기 때문입니다. 부드러운 보간을 적용하면 이 딜레이와 노이즈를 최소화하면서 어느 정도 실시간 데이터를 추적할 수 있게 해 줍니다. 식은 다음과 같습니다:
이 식에서:
self.prev_pose3d는 이전 프레임에서의 Vive Tracker 위치(Position) 값입니다.
pose3d는 현재 프레임에서의 Vive Tracker 위치(Position) 값입니다.
params.additional_smoothing는 사용자가 설정할 수 있는 부드러운 보간의 가중치로 0과 1 사이의 값을 갖습니다.
self.prev_pose3d * params.additional_smoothing 부분은 이전 프레임에서의 위치 값을 완화된 가중치로 곱하여 사용합니다. 
반대로, pose3d * (1 - params.additional_smoothing) 부분은 현재 프레임에서의 위치 값을 (1 - params.additional_smoothing) 가중치로 곱하여 사용합니다. 
이 두 값의 합으로 새로운 위치를 계산하여 두 프레임의 위치 데이터를 부드럽게 보간합니다.
이렇게 가중치에 따라 이전 프레임과 현재 프레임의 위치 데이터를 섞어 이동이 부드럽게 이루어지는 것을 확인할 수 있습니다. 
가중치 값이 클수록 이전 프레임의 위치 데이터가 더 많이 반영되며, 가중치 값이 작을수록 현재 프레임의 위치 데이터가 더 많이 반영됩니다.
          '''
        headsetpos = [float(0),float(0),float(0)]
        headsetrot = R.from_quat([float(0),float(0),float(0),float(1)])
        ###headsetpos를 [0, 0, 0]으로 초기화하고, headsetrot를 [0, 0, 0, 1] 즉, 초기화된 identity(항등) 쿼터니언(quaternion)으로 생성합니다.

        '''
쿼터니언(quaternion)은 3D 공간에서 자유롭게 회전할 수 있는 도구로 사용되며, 각 축(x, y, z)을 기준으로 회전할 실수 파트와 스칼라 파트를 결합하여 생성됩니다. 쿼터니언의 초기 형태는 [0, 0, 0, 1]입니다. 이때, [x, y, z] 부분은 벡터 부분으로 회전 축을 나타내고, 마지막 원소 1은 스칼라 부분으로 회전의 각도를 나타낸다고 볼 수 있습니다. 설정된 값이 [0, 0, 0, 1]인 경우, 회전 축이 없고 스칼라 부분이 회전 각도 0을 의미하므로 회전이 없는 초기 상태를 나타낸다고 할 수 있습니다. 이를 identity(항등) 쿼터니언이라고 합니다. headsetrot = R.from_quat([0, 0, 0, 1])에서 [0, 0, 0, 1]은 회전이 없는 초기 상태를 나타내는 값입니다. 쿼터니언 [0, 0, 0, 0]은 존재하지 않는 회전을 나타냅니다. 일반적으로 쿼터니언의 항등 요소(identity element)는 [0, 0, 0, 1]이므로, 회전을 초기화할 때 이 값을 사용합니다.
    '''
        neckoffset = headsetrot.apply(params.hmd_to_neck_offset)   #the neck position seems to be the best point to allign to, as its well defined on
                                                            #the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.
        ###neckoffset 값은 headsetrot을 기존의 params.hmd_to_neck_offset 벡터에 적용한 neck position 값을 계산합니다.
      
        if params.recalibrate:
            print("frame to recalibrate")
            ###params.recalibrate 값이 True 이면 다시 보정해야 함을 나타내는 문구를 출력하고, 그렇지 않으면 아래 코드를 실행합니다.

        else:
            pose3d = pose3d * params.posescale     #rescale skeleton to calibrated height
            #print(pose3d)
            ###skeleton의 위치 값을 보정된 높이로 조절하기 위해 params.posescale 값으로 pose3d 값을 곱합니다.
            
            offset = pose3d[7] - (headsetpos+neckoffset)    #calculate the position of the skeleton
            ### 스켈레톤의 위치를 계산하기 위해서 pose3d[7]에서 헤드셋의 위치 headsetpos와 넥 위치(neck position)의 오프셋 neckoffset을 더한 값을 뺍니다.
            
            if not params.preview_skeleton:
                ###params.preview_skeleton이 False일 때, 스켈레톤의 트래커 데이터를 생성하고, 위치와 회전값을 조정한 뒤, 이 정보를 OSC 메시지로 보내는 데 사용합니다. 그렇지 않으면, else 부분의 코드를 실행합니다.

                trackers = []
                trackers.append({ "name": "head", "position": [ 0, 0, 0 ]})
                ###trackers라는 리스트에 트래커 데이터를 저장하기 위한 초기 설정을 합니다. 
                # 헤드 트래커의 경우 "head"이름과 위치 [0, 0, 0] 정보를 추가합니다. 
                # 아래 코드는 공통적인 트래커 정보 생성 및 조정을 실행하는 반복문입니다. 첫 번째 반복문의 경우 고정 허리를 무시하지 않고,
                # 두 번째 경우에는 고정 허리를 무시하고 작동합니다.

                if not params.ignore_hip:
                    for i in [(0,1),(5,2),(6,0)]:
                        ###사용될 각 코드 조각에서 트래커의 위치 값에 대해 스켈레톤 위치에 대한 오프셋을 적용합니다.
                        #left foot should be position 0 and rotation 1, but for osc, the rotations got switched at some point so its (0,2)
                        position = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        #position[0] = -position[0]
                        #position[1] = -position[1]
                        position[2] = -position[2]
                        ###슬롯의 회전값을 적절한 회전 값으로 조절하여 변환하고, 좌우 반전(거울 이미지)을 적용합니다:


                        rotation = R.from_quat(rots[i[1]])
                        #rotation *= R.from_euler("ZY", [ 180, -90 ], degrees=True)
                        rotation = rotation.as_euler("zxy", degrees=True)
                        rotation = [ -rotation[1], -rotation[2], rotation[0] ]  #mirror the rotation, as we mirrored the positions
                        ###그런 다음 trackers 리스트에 이름, 위치 및 회전 데이터를 포함하는 트래커 객체를 추가합니다.

                        trackers.append({ "name": str(i[1]+1), "position": position, "rotation": rotation })
                        ###위의 반복문에서, 고정 허리 무시 여부에 따라 각각 좌표 및 회전 값이 저장되고, 최종 저장 정보로 OSC 메시지에 전달되어 데이터를 스켈레톤에 적용하는데 활용됩니다.

                        
                else:
                    for i in [(0,1),(5,2)]:
                        position = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        #position[0] = -position[0]
                        #position[1] = -position[1]
                        position[2] = -position[2]
                        rotation = R.from_quat(rots[i[1]])
                        #rotation *= R.from_euler("ZY", [ 180, -90 ], degrees=True)
                        rotation = rotation.as_euler("zxy", degrees=True)
                        rotation = [ -rotation[1], -rotation[2], rotation[0] ]
                        trackers.append({ "name": str(i[1]+1), "position": position, "rotation": rotation })
                if params.use_hands:
                    # Sending hand trackers unsupported
                    pass
                if len(trackers) > 1:
                    self.client.send(osc_build_bundle(trackers))

            else:
                # Preview skeleton unsupported
                pass
        return True

    def disconnect(self):
        pass
