import time
"""
time:

time은 시간과 관련된 기능을 제공하는 파이썬 표준 라이브러리입니다.
시간에 대한 정보를 얻고, 시간 간격을 측정하거나 지연을 생성하는 데 사용할 수 있습니다.

"""
import socket
"""
TCP/IP 기반의 소켓 통신을 구현할 수 있으며, 서버와 클라이언트 간의 데이터 전송을 가능하게 합니다.
주요 함수로는 소켓 생성 및 연결을 위한 socket(), 데이터 송수신을 위한 send(), recv(), 연결 종료를 위한 close() 등이 있습니다.
"""
import cv2
"""
cv2 (OpenCV):

cv2는 영상 처리와 컴퓨터 비전을 위한 라이브러리인 OpenCV(Open Source Computer Vision Library)의 파이썬 인터페이스입니다.
영상 데이터를 읽고 쓰기, 영상 처리 및 분석, 객체 검출 및 추적, 카메라 캡처 등 다양한 컴퓨터 비전 작업을 수행할 수 있습니다.
주요 함수로는 영상 파일 읽기/쓰기를 위한 imread(), imwrite(), 이미지 변환을 위한 cvtColor(), 이미지 표시를 위한 imshow() 등이 있습니다.
"""
import numpy as np 
"""
numpy:

numpy는 다차원 배열과 행렬 연산을 위한 라이브러리입니다.
파이썬에서 과학적이고 수치적인 연산을 수행할 수 있게 해주며, 데이터 구조와 연산 함수를 제공합니다.
주요 기능으로는 배열 생성, 형태 변경, 슬라이싱, 인덱싱, 수학 연산, 선형 대수, 푸리에 변환, 랜덤 값 생성 등이 있습니다.
"""
from sys import platform
"""

sys:

sys는 파이썬 인터프리터와 관련된 기능을 제공하는 모듈입니다.
명령 행 인수, 스크립트 종료, 표준 입출력, 예외 처리 등과 같은 기능을 사용할 수 있습니다.
주요 요소로는 명령 행 인수를 얻기 위한 argv, 스크립트 종료를 위한 exit, 표준 입출력을 위한 stdin, stdout, stderr 등이 있습니다.
"""
from scipy.spatial.transform import Rotation as R
"""
scipy는 과학 및 수학 연산을 위한 라이브러리입니다.
scipy.spatial.transform.Rotation은 회전을 표현하고 조작하는 클래스입니다.
회전 표현 형식을 변환하거나 회전 연산을 수행하는 등의 기능을 제공합니다.
"""
import cv2
import threading
"""
threading:

threading은 스레드 기반 병렬 처리를 지원하는 라이브러리입니다.
여러 개의 스레드를 생성하고 제어하여 동시에 실행되는 작업을 처리할 수 있습니다.
주요 클래스로는 Thread, Lock, Condition, Semaphore 등이 있으며, 이를 활용하여 스레드 간의 동기화와 통신을 할 수 있습니다.
"""
import sys

def draw_pose(frame,pose,size):
    pose = pose*size
    for sk in EDGES: #sk는 EDGES에서 한 쌍의 관절 지점 인덱스를 가져와서 해당 관절 지점을 연결하는 선을 그리는 데 사용되는 변수입니다.
        cv2.line(frame,(int(pose[sk[0],1]),int(pose[sk[0],0])),(int(pose[sk[1],1]),int(pose[sk[1],0])),(0,255,0),3)


"""
이 코드에서 frame, pose, size는 다음과 같은 변수들을 나타냅니다:

frame: 이미지나 프레임을 나타내는 변수입니다. OpenCV에서는 이미지나 프레임을 다차원 배열로 표현하며,
이 변수에 해당 배열을 전달하여 그림을 그릴 대상 이미지를 지정합니다.

   이 코드에서는 frame에 그림을 그릴 대상 이미지가 전달됩니다.

pose: 포즈(Pose)를 나타내는 변수입니다. 포즈는 사람이나 객체의 자세를 나타내는 정보로, 
일반적으로 관절들의 위치를 포함한 데이터입니다.

pose 변수는 2차원 배열로 표현되며,각 관절의 위치를 좌표로 표현합니다. 

예를 들어, pose[i]는 i번째 관절의 위치를 나타내는 좌표 [x, y]를 가지고 있습니다.

size: 포즈의 크기(scale)를 조절하기 위한 변수입니다.

 이 값은 포즈를 그릴 때 사용됩니다. 
 보통 이미지나 프레임의 크기에 따라 다르게 설정될 수 있습니다. 

 예를 들어, 포즈를 원본 이미지의 크기의 절반 크기로 표현하고 싶다면, size는 0.5가 될 수 있습니다.

이 함수는 주어진 pose를 이용하여 frame에 포즈를 그리는 역할을 합니다. 

pose에 있는 관절들의 위치를 size를 적용하여 스케일을 조정하고, 그려진 포즈를 frame에 선으로 그립니다.

 EDGES는 관절을 이어 그릴 때 사용되는 선들의 정보를 담고 있는 변수로, 
 이 코드에서는 이 정보를 이용하여 관절들을 선으로 연결합니다.

1. 이 함수는 2D 스켈레톤을 이미지(frame) 위에 그리는데 사용됩니다. 

입력값으로 이미지(frame), 스켈레톤 자세 데이터(pose), 그리기를 원하는 크기(size)가 필요합니다.

2. 코드 한 줄씩 분석:

    - `pose = pose * size`: 입력으로 주어진 스켈레톤 자세 데이터의 좌표(pose)에 크기(size)를 곱하여 크기를 조절합니다.
    - `for sk in EDGES:`: 스켈레톤의 개별 부분(간선)을 순회하기 위한 for문입니다. 각 개별 부분을 조사하고 선을 그립니다.
    - 내부의 코드인 `cv2.line(frame, (int(pose[sk[0], 1]), int(pose[sk[0], 0])), (int(pose[sk[1], 1]), int(pose[sk[1], 0])), (0, 255, 0), 3)`: 
    이미지(frame) 위에 각 스켈레톤 부분에 대한 선을 그립니다. 그리는 좌표는 스켈레톤 부분의 시작점 `pose[sk[0]]`과 끝점 `pose[sk[1]]`을 사용하며, 
   !!!!!!!!!!!!! 선 색상은 녹색(`(0, 255, 0)`), 그리고 선 두께는 3으로 설정되어 있습니다.!!!!!!!!!!!

이렇게 `draw_pose` 함수는 프레임 위에 스켈레톤 자세 데이터를 그립니다. 

"""


def mediapipeTo3dpose(lms):
    #33 pose landmarks as in https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
    #convert landmarks returned by mediapipe to skeleton that I use.
    #lms = results.pose_world_landmarks.landmark
    
    pose = np.zeros((29,3))
    """
    np.zeros((29,3))는 NumPy 라이브러리의 함수인 zeros()를 사용하여 0으로 초기화된 29x3 크기의 2차원 배열을 생성하는 것을 의미합니다.

    NumPy는 파이썬에서 수치 계산을 위한 강력한 라이브러리로, 다차원 배열을 효율적으로 처리할 수 있는 기능을 제공합니다. 
    zeros() 함수는 주어진 크기의 배열을 생성하고 모든 요소를 0으로 채웁니다. 
   위의 코드에서는 29개의 행과 3개의 열을 가진 배열을 생성하고, 모든 요소를 0으로 초기화하여 pose라는 변수에 할당합니다.

   이렇게 초기화된 배열은 나중에 랜드마크의 위치 정보를 저장하는 용도로 사용됩니다.
   초기값인 0은 아직 채워지지 않은 포즈 스켈레톤의 위치를 나타냅니다. 이후 코드에서 해당 위치에 실제 랜드마크의 좌표 값을 할당하여 포즈 스켈레톤을 완성시킵니다.
    """
    """
    lms는 사람의 특정한 관절위치의 변수임
    """
    pose[0]=[lms[28].x,lms[28].y,lms[28].z]  # 첫 번째 행에는 lms[28]의 x, y, z 좌표 값을 할당합니다.
    pose[1]=[lms[26].x,lms[26].y,lms[26].z]  # 두 번째 행에는 lms[26]의 x, y, z 좌표 값을 할당합니다.
    pose[2]=[lms[24].x,lms[24].y,lms[24].z]  # 세 번째 행에는 lms[24]의 x, y, z 좌표 값을 할당합니다.
    pose[3]=[lms[23].x,lms[23].y,lms[23].z]  # 네 번째 행에는 lms[23]의 x, y, z 좌표 값을 할당합니다.
    pose[4]=[lms[25].x,lms[25].y,lms[25].z]  # 다섯 번째 행에는 lms[25]의 x, y, z 좌표 값을 할당합니다.
    pose[5]=[lms[27].x,lms[27].y,lms[27].z]  # 여섯 번째 행에는 lms[27]의 x, y, z 좌표 값을 할당합니다.
    pose[6]=[0,0,0]  # 일곱 번째 행은 [0, 0, 0]으로 초기화합니다.
    #some keypoints in mediapipe are missing, so we calculate them as avarage of two keypoints

    pose[7]=[lms[12].x/2+lms[11].x/2,lms[12].y/2+lms[11].y/2,lms[12].z/2+lms[11].z/2]  # 여덟 번째 행에는 lms[12]와 lms[11]의 x, y, z 좌표 값의 평균을 할당합니다.
    pose[8]=[lms[10].x/2+lms[9].x/2,lms[10].y/2+lms[9].y/2,lms[10].z/2+lms[9].z/2]  # 아홉 번째 행에는 lms[10]과 lms[9]의 x, y, z 좌표 값의 평균을 할당합니다.

    pose[9]=[lms[0].x,lms[0].y,lms[0].z]  # 열 번째 행에는 lms[0]의 x, y, z 좌표 값을 할당합니다.
    pose[10]=[lms[15].x,lms[15].y,lms[15].z]  # 열한 번째 행에는 lms[15]의 x, y, z 좌표 값을 할당합니다.
    pose[11]=[lms[13].x,lms[13].y,lms[13].z]  # 열두 번째 행에는 lms[13]의 x, y, z 좌표 값을 할당합니다.
    pose[12]=[lms[11].x,lms[11].y,lms[11].z]  # 열세 번째 행에는 lms[11]의 x, y, z 좌표 값을 할당합니다.
    pose[13]=[lms[12].x,lms[12].y,lms[12].z]  # 열네 번째 행에는 lms[12]의 x, y, z 좌표 값을 할당합니다.
    pose[14]=[lms[14].x,lms[14].y,lms[14].z]  # 열다섯 번째 행에는 lms[14]의 x, y, z 좌표 값을 할당합니다.
    pose[15]=[lms[16].x,lms[16].y,lms[16].z]  # 열여섯 번째 행에는 lms[16]의 x, y, z 좌표 값을 할당합니다.
    
    pose[16]=[pose[6][0]/2+pose[7][0]/2,pose[6][1]/2+pose[7][1]/2,pose[6][2]/2+pose[7][2]/2]  # 열일곱 번째 행에는 pose[6]과 pose[7]의 x, y, z 좌표 값의 평균을 할당합니다.

    pose[17] = [lms[31].x,lms[31].y,lms[31].z]  # 열여덟 번째 행에는 lms[31]의 x, y, z 좌표 값을 할당합니다. (오른쪽 발 전방)
    pose[18] = [lms[29].x,lms[29].y,lms[29].z]  # 열아홉 번째 행에는 lms[29]의 x, y, z 좌표 값을 할당합니다. (오른쪽 발 후방)
    pose[19] = [lms[25].x,lms[25].y,lms[25].z]  # 스물 번째 행에는 lms[25]의 x, y, z 좌표 값을 할당합니다. (오른쪽 발 위)

    pose[20] = [lms[32].x,lms[32].y,lms[32].z]  # 스물한 번째 행에는 lms[32]의 x, y, z 좌표 값을 할당합니다. (왼쪽 발 전방)
    pose[21] = [lms[30].x,lms[30].y,lms[30].z]  # 스물두 번째 행에는 lms[30]의 x, y, z 좌표 값을 할당합니다. (왼쪽 발 후방)
    pose[22] = [lms[26].x,lms[26].y,lms[26].z]  # 스물세 번째 행에는 lms[26]의 x, y, z 좌표 값을 할당합니다. (왼쪽 발 위)

    pose[23] = [lms[17].x,lms[17].y,lms[17].z]  # 스물네 번째 행에는 lms[17]의 x, y, z 좌표 값을 할당합니다. (오른쪽 손 전방)
    pose[24] = [lms[15].x,lms[15].y,lms[15].z]  # 스물다섯 번째 행에는 lms[15]의 x, y, z 좌표 값을 할당합니다. (오른쪽 손 후방)
    pose[25] = [lms[19].x,lms[19].y,lms[19].z]  # 스물여섯 번째 행에는 lms[19]의 x, y, z 좌표 값을 할당합니다. (오른쪽 손 위)

    pose[26] = [lms[18].x,lms[18].y,lms[18].z]  # 스물일곱 번째 행에는 lms[18]의 x, y, z 좌표 값을 할당합니다. (왼쪽 손 전방)
    pose[27] = [lms[16].x,lms[16].y,lms[16].z]  # 스물여덟 번째 행에는 lms[16]의 x, y, z 좌표 값을 할당합니다. (왼쪽 손 후방)
    pose[28] = [lms[20].x,lms[20].y,lms[20].z]  # 스물아홉 번째 행에는 lms[20]의 x, y, z 좌표 값을 할당합니다. (왼쪽 손 위)

    return pose  # 최종적으로 구성된 pose 배열을 반환합니다.


"""

파이썬에서 '랜드마크'는 일반적으로 얼굴 인식이나 객체 인식과 관련하여 사용되는 용어입니다.

 랜드마크는 얼굴이나 객체의 특정 부분을 나타내는 지점이나 위치를 말합니다. 이러한 랜드마크는 해당 얼굴이나 객체의 특징을 식별하고 분석하는 데 사용됩니다.

예를 들어, 얼굴 인식에서 얼굴의 랜드마크는 눈, 코, 입 등과 같은 특정 부위의 위치를 가리킵니다. 

얼굴 랜드마크는 얼굴 인식을 통해 얼굴의 특징을 식별하고 분석하는 데 사용되며, 표정 인식, 감정 분석, 안경 착용 여부 등 다양한 얼굴 관련 작업에 활용될 수 있습니다.

객체 인식에서도 랜드마크는 객체의 특정 부분을 나타내는 지점이 될 수 있습니다. 예를 들어, 사람 인식에서는 사람의 어깨, 손, 발 등이 랜드마크가 될 수 있습니다. 

이러한 랜드마크 정보를 활용하여 객체의 위치, 크기, 방향 등을 추정하거나 객체 간의 상호작용을 분석할 수 있습니다.


1. 함수 인자로 랜드마크 데이터 `lms`를 받습니다. 이 데이터는 3D 포즈 랜드마크로 구성되어 있습니다. -> OPENCV오픈소스 참고해야함 

2. 3D 포즈를 표현하기 위해 (29,3) 모양의 NumPy 배열 `pose`를 초기화합니다. 각 행은 3D 좌표(x, y, z)를 가지며, 해당 좌표의 재구성을 위한 29개의 랜드마크가 있습니다.

3. 각 랜드마크를 대응하는 배열 인덱스로 할당합니다. 예를 들어, 첫 번째 랜드마크는 `lms[28]`에 저장되어 있으며 `pose[0]`으로 할당됩니다. 각 좌표는 x, y, z 형식으로 다루어집니다.

4. 일부 랜드마크는 Mediapipe에서 누락되기 때문에 이들은 주변 랜드마크의 평균값으로 계산됩니다. 예를 들어, `pose[7]`은 `lms[12]`와 `lms[11]`의 평균으로 설정됩니다. 
이렇게 평균화된 좌표는 x, y, z에 대해 각각 속하는 값들을 균등하게 나눕니다.

5. 함수가 끝나면 변환된 3D 포즈 `pose`를 반환합니다.

"""

def keypoints_to_original(scale,center,points):
    scores = points[:,2] #`scores = points[:,2]`에서 points의 세 번째 열(인덱스 2)을 `scores` 변수에 저장합니다. 이 값들은 각 점의 신뢰도를 나타내는 것입니다.
    points -= 0.5# `points -= 0.5`에서 모든 포인트들에 0.5를 빼줍니다. 이렇게 함으로써 각 좌표의 범위를 -0.5 ~ 0.5로 조정합니다.

    points *= scale#`points *= scale`에서 각 point에 스케일(scale)을 곱합니다. 이렇게 하여 해당 포인트들의 크기를 조절합니다.
    points[:,0] += center[0]#네 번째와 다섯 번째 줄(`points[:,0] += center[0]`와 `points[:,1] += center[1]`)에서 center의 좌표를 points에 더해줍니다. 
    points[:,1] += center[1]#이렇게 하여 이미지 내에서 포인트들의 위치를 조정합니다.
    
    points[:,2] = scores#`points[:,2] = scores`에서 원래 신뢰도(점수) 값을 각 포인트의 세 번째 열에 다시 저장합니다.
    
    return points#`return points`에서 스케일과 위치가 조정된 포인트를 반환합니다.


"""
이 함수에서 scale, center, points는 다음과 같은 변수들을 나타냅니다:

scale: 랜드마크 포인트들의 크기(scale)를 나타내는 값입니다. 

이 값은 랜드마크 포인트들을 변환할 때 사용됩니다. 보통 이미지나 프레임의 크기에 따라 다르게 설정될 수 있습니다.

 예를 들어, 랜드마크 포인트들이 원본 이미지 크기의 절반 크기로 표현되어 있다면, scale은 0.5가 될 수 있습니다.


center: 랜드마크 포인트들의 중심(center) 위치를 나타내는 값입니다.

 이 값은 랜드마크 포인트들을 변환할 때 사용됩니다. 예를 들어, 이미지의 중심 좌표가 (x_center, y_center)일 때, 

 center는 [x_center, y_center]와 같이 표현될 수 있습니다. 이렇게 설정된 center를 기준으로 랜드마크 포인트들이 이동하게 됩니다.

points: 랜드마크 포인트들을 나타내는 배열입니다. 
이 배열은 2차원 배열로, 각 랜드마크 포인트의 좌표와 추가적인 정보를 포함합니다.

 주로 [x, y, score]와 같은 형태로 표현됩니다. 여기서 x와 y는 랜드마크 포인트의 위치 좌표이고, score는 해당 랜드마크의 신뢰도나 중요도를 나타내는 값입니다.

   이 함수에서는 랜드마크 포인트들을 원본 크기와 중심을 기준으로 스케일(scale)과 이동(center)을 적용한 후, 각 포인트의 신뢰도(score) 값을 유지하도록 조정합니다.

따라서 이 함수는 주어진 scale과 center를 이용하여 points 배열의 랜드마크 포인트들을 원본 이미지나 프레임에 대한 좌표로 변환해주는 역할을 합니다.
변환된 랜드마크 포인트 배열이 반환됩니다.


"""
def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2 # 입력 array X의 마지막 차원이 2인지 확인합니다. 이를 통해 X가 2차원 점들로 구성되어 있음을 보장합니다.
    """
    assert 문은 주로 디버깅과 테스트 목적으로 사용됩니다. 프로그램의 실행 중에 특정 조건이 만족되어야 하는지 확인하고,
    그렇지 않으면 프로그램을 중단하고 오류를 식별하는 데 도움이 됩니다.
    예를 들어, 함수의 입력 값이 특정 범위에 속하는지 확인하거나, 함수의 반환 값이 예상한 결과와 일치하는지 확인하는 데 사용될 수 있습니다.
    """

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w] 
"""
`return X / w * 2 - [1, h / w]`: X의 좌표를 정규화하는 부분입니다. 여기서 w와 h는 너비와 높이입니다. 각 좌표에 대해 다음과 같은 변환을 수행합니다:
   - X의 각 원소를 너비(w)로 나눕니다.
   - 2를 곱하여 범위를 [0, 2]로 조정합니다.
   - [1, h / w] 벡터를 빼주어 x 좌표의 범위를 [-1, 1]로, y 좌표의 범위를 [0, h/w]의 비율을 유지하면서 -1에서 1 사이로 조정합니다.

    함수는 주어진 점들의 좌표 값을 화면 크기에 적절하게 조절하기 위한 정규화 작업을 수행합니다. 
    이렇게 하여 동일한 화면 크기에서 다양한 크기의 영상 소스에 공통적으로 적용할 수 있습니다.
"""

"""
이 코드에서 X, w, h는 다음과 같은 변수들을 나타냅니다:

X: 화면 좌표를 나타내는 변수입니다. X는 2차원 배열로 표현되며, 각 점의 x와 y 좌표를 가지고 있습니다. 예를 들어, X[i]는 i번째 점의 좌표 [x, y]를 가지고 있습니다.

w: 화면의 가로 너비를 나타내는 변수입니다. w는 숫자로 표현되며, 화면의 가로 크기를 나타냅니다.

h: 화면의 세로 높이를 나타내는 변수입니다. h는 숫자로 표현되며, 화면의 세로 크기를 나타냅니다.

이 함수는 주어진 화면 좌표 X를 정규화된 화면 좌표로 변환하는 역할을 합니다. 변환은 다음과 같은 단계로 이루어집니다:

화면 좌표 정규화: X를 입력으로 받은 화면의 가로 너비 w로 나누어서 [0, w] 범위의 좌표를 [0, 1] 범위로 정규화합니다.

정규화된 화면 좌표 변환: 정규화된 좌표를 [-1, 1] 범위로 변환합니다. 이 때, 세로 높이 h를 가로 너비 w로 나누어서 비율을 유지합니다.

변환된 정규화 화면 좌표 반환: 변환된 정규화 화면 좌표를 반환합니다.

즉, 이 함수는 주어진 화면 좌표를 입력으로 받아서 해당 화면의 가로 너비와 세로 높이에 맞게 정규화된 화면 좌표로 변환합니다. 
이러한 변환은 보통 화면 상의 좌표를 다룰 때 사용된다.

"""

def get_rot_hands(pose3d):
#이 함수는 손의 3차원 자세(3D pose)를 입력받아 왼손과 오른손의 회전을 반환합니다.

#이 코드에서 pose3d는 3D 포즈 정보를 나타내는 변수입니다. pose3d는 배열 또는 리스트 형태로 주어지며, 각 원소는 3D 공간에서의 점의 좌표를 나타냅니다. -> 관절의 좌표를 나타냄
    hand_r_f = pose3d[26]
    hand_r_b = pose3d[27]
    hand_r_u = pose3d[28]
    
    hand_l_f = pose3d[23]
    hand_l_b = pose3d[24]
    hand_l_u = pose3d[25]
    #이 부분에서는 각 손의 앞쪽, 뒤쪽 및 위쪽 벡터 좌표를 입력 3D 자세에서 추출합니다.

    # left hand
    
    x = hand_l_f - hand_l_b #왼손의 앞쪽 벡터(x축 방향)은 왼손 앞 지점(hand_l_f)과 왼손 뒷 지점(hand_l_b)의 차이입니다.
    w = hand_l_u - hand_l_b #왼손의 위쪽 벡터(y축 방향)은 왼손 위 지점(hand_l_u)과 왼손 뒷 지점(hand_l_b)의 차이입니다.
    z = np.cross(x, w)#왼손의 앞쪽 벡터(x)와 위쪽 벡터(w)의 외적을 통해 왼손의 오른쪽 방향 벡터(z축 방향)을 구합니다.
    y = np.cross(z, x)#오른쪽 방향 벡터(z)와 앞쪽 벡터(x)의 외적을 통해 왼손의 위쪽 방향 벡터(y축 방향)을 구합니다.
    
    x = x/np.sqrt(sum(x**2))#앞쪽 벡터(x)를 크기로 나누어 정규화합니다.
    y = y/np.sqrt(sum(y**2))#위쪽 벡터(y)를 크기로 나누어 정규화합니다.
    z = z/np.sqrt(sum(z**2))#오른쪽 방향 벡터(z)를 크기로 나누어 정규화합니다
    
    l_hand_rot = np.vstack((z, y, -x)).T#
    
    # right hand
    
    x = hand_r_f - hand_r_b
    w = hand_r_u - hand_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    r_hand_rot = np.vstack((z, y, -x)).T
    """
    오른손의 앞쪽, 위쪽, 오른쪽 방향 벡터를 행렬로 구성하여 오른손의 로컬 좌표계를 표현하는 행렬을 생성합니다. 
    z는 앞쪽 방향 벡터, y는 위쪽 방향 벡터, -x는 오른쪽 방향 벡터입니다. 
    vstack() 함수를 사용하여 이 벡터들을 수직으로 쌓은 다음, .T를 사용하여 행과 열을 바꿔 행렬을 전치합니다.
    """

    r_hand_rot = R.from_matrix(r_hand_rot).as_quat()
    """
    생성된 오른손의 로컬 좌표계 행렬을 회전 객체로 변환합니다. R.from_matrix() 함수는 행렬을 회전 객체로 변환하는데 사용됩니다. 
    이후 .as_quat() 메서드를 호출하여 회전 객체를 쿼터니언(quaternion)으로 변환합니다. 
    최종적으로 오른손의 회전을 나타내는 쿼터니언 값을 얻습니다.
    """
    l_hand_rot = R.from_matrix(l_hand_rot).as_quat()
    """
    동일한 과정을 왼손에 대해서 수행합니다.
    왼손의 로컬 좌표계 행렬을 회전 객체로 변환하고, 쿼터니언 값으로 변환합니다.
    """
    
    return l_hand_rot, r_hand_rot#왼손과 오른손의 회전을 나타내는 쿼터니언 값을 반환합니다.

"""
쿼터니언(Quaternion)은 3D 공간에서 회전을 나타내는 수학적인 개념입니다. 쿼터니언은 네 개의 실수로 구성된 벡터로 표현되며, 일반적으로 (w, x, y, z) 형태로 표기됩니다.

쿼터니언은 오일러 각(euler angles)이나 회전 행렬(rotation matrix)보다 효율적인 회전 표현 방식입니다. 
쿼터니언은 회전을 3차원 공간에서 4차원으로 확장하여 나타내기 때문에 표현력이 높으며, 회전 연산을 효율적으로 수행할 수 있습니다. 
또한, 쿼터니언은 회전 중심, 회전 각도, 회전 축 등 다양한 회전 관련 연산을 수행하는 데 유용합니다.

특히, 자세 추정에서는 쿼터니언을 사용하여 객체나 인간의 자세를 정확하게 표현하고 추적할 수 있습니다.

위의 코드에서 R.from_matrix().as_quat()는 회전 행렬로부터 쿼터니언으로 변환하는 과정입니다. 이렇게 변환된 쿼터니언 값은 3D 자세의 회전을 나타내는데 사용됩니다.
"""

"""

이 코드에서 pose3d는 3D 포즈 정보를 나타내는 변수입니다. pose3d는 배열 또는 리스트 형태로 주어지며, 각 원소는 3D 공간에서의 점의 좌표를 나타냅니다.

이 코드는 주어진 pose3d에서 왼손과 오른손의 회전 정보를 추출하는 함수입니다.

 코드 내에서는 pose3d 배열에서 특정 인덱스에 해당하는 값들을 사용하여 회전을 계산합니다. 

 왼손과 오른손 각각의 앞, 뒤, 위 점을 선택한 후, 벡터 연산을 통해 회전에 필요한 축을 계산합니다.

왼손과 오른손의 회전을 계산한 후, R.from_matrix() 함수를 사용하여 회전 행렬로 변환하고, as_quat() 메서드를 사용하여 쿼터니언(quaternion) 형태로 변환합니다. 

최종적으로 왼손과 오른손의 회전 쿼터니언을 반환합니다.

따라서, pose3d는 3D 포즈 정보를 담고 있는 변수이며, 주로 관절의 좌표를 포함합니다.


"""

def get_rot_mediapipe(pose3d):
    #이 함수는 힙, 왼쪽 발, 오른쪽 발의 3차원 자세(3D pose)를 사용하여 회전을 계산하고 반환합니다.
    hip_left = pose3d[2]
    hip_right = pose3d[3]
    hip_up = pose3d[16]
    #주어진 pose3d 배열에서 엉덩이의 왼쪽, 오른쪽, 위쪽 좌표를 선택하여 변수에 할당합니다.
    foot_l_f = pose3d[20]
    foot_l_b = pose3d[21]
    foot_l_u = pose3d[22]
    #주어진 pose3d 배열에서 왼쪽 발의 앞, 뒤, 위 좌표를 선택하여 변수에 할당합니다.
    foot_r_f = pose3d[17]
    foot_r_b = pose3d[18]
    foot_r_u = pose3d[19]
    #주어진 pose3d 배열에서 오른쪽 발의 앞, 뒤, 위 좌표를 선택하여 변수에 할당합니다.

    # hip
    
    x = hip_right - hip_left
    w = hip_up - hip_left
    z = np.cross(x, w)
    y = np.cross(z, x)
    """
    엉덩이에서 발로 향하는 벡터(x)와 위로 향하는 벡터(w)를 계산합니다.
    x는 hip_right - hip_left로 구하고, w는 hip_up - hip_left로 구합니다. 
    그런 다음 np.cross() 함수를 사용하여 x와 w의 외적을 계산하여 수직인 벡터 z를 구하고, 다시 z와 x의 외적을 계산하여 수직인 벡터 y를 구합니다.
    """
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    #x, y, z 벡터들을 정규화(normalize)합니다. 각 벡터의 크기를 해당 벡터의 제곱 합의 제곱근으로 나누어 각 벡터의 크기를 1로 만듭니다.
    hip_rot = np.vstack((x, y, z)).T
    #x, y, z 벡터들을 수직으로 쌓아서 행렬로 변환합니다. np.vstack() 함수를 사용하여 열 벡터들을 수직으로 쌓고, .T를 사용하여 행렬을 전치합니다.

    # left foot
    
    x = foot_l_f - foot_l_b
    w = foot_l_u - foot_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    """
    왼쪽 발에서 앞으로 향하는 벡터(x)와 위로 향하는 벡터(w)를 계산합니다. x는 foot_l_f - foot_l_b로 구하고, w는 foot_l_u - foot_l_b로 구합니다. 
    그런 다음 np.cross() 함수를 사용하여 x와 w의 외적을 계산하여 수직인 벡터 z를 구하고, 다시 z와 x의 외적을 계산하여 수직인 벡터 y를 구합니다.
    """
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
   #x, y, z 벡터들을 정규화합니다. 각 벡터의 크기를 해당 벡터의 제곱 합의 제곱근으로 나누어 각 벡터의 크기를 1로 만듭니다.
    l_foot_rot = np.vstack((x, y, z)).T
    
    # x, y, z 벡터들을 수직으로 쌓아서 행렬로 변환합니다. np.vstack() 함수를 사용하여 열 벡터들을 수직으로 쌓고, .T를 사용하여 행렬을 전치합니다.
    
    # right foot
    
    x = foot_r_f - foot_r_b
    w = foot_r_u - foot_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    """
    오른쪽 발에서 앞으로 향하는 벡터(x)와 위로 향하는 벡터(w)를 계산합니다. x는 foot_r_f - foot_r_b로 구하고, w는 foot_r_u - foot_r_b로 구합니다.
      그런 다음 np.cross() 함수를 사용하여 x와 w의 외적을 계산하여 수직인 벡터 z를 구하고, 다시 z와 x의 외적을 계산하여 수직인 벡터 y를 구합니다.
    """
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
   #x, y, z 벡터들을 정규화합니다. 각 벡터의 크기를 해당 벡터의 제곱 합의 제곱근으로 나누어 각 벡터의 크기를 1로 만듭니다.
    r_foot_rot = np.vstack((x, y, z)).T
   #x, y, z 벡터들을 수직으로 쌓아서 행렬로 변환합니다. np.vstack() 함수를 사용하여 열 벡터들을 수직으로 쌓고, .T를 사용하여 행렬을 전치합니다.
    hip_rot = R.from_matrix(hip_rot).as_quat()
    r_foot_rot = R.from_matrix(r_foot_rot).as_quat()
    l_foot_rot = R.from_matrix(l_foot_rot).as_quat()
    
    #hip_rot, r_foot_rot, l_foot_rot 행렬을 R.from_matrix() 함수를 사용하여 회전 행렬로 변환하고, .as_quat() 메서드를 사용하여 각각의 회전을 쿼터니언 형태로 변환합니다.
   
    return hip_rot, l_foot_rot, r_foot_rot # 엉덩이의 회전 쿼터니언과 왼쪽 발, 오른쪽 발의 회전 쿼터니언을 반환합니다.

"""
계산된 회전은 Quaternion 형태로 변환한 다음 힙, 왼쪽 발, 오른쪽 발 회전으로 반환합니다.

"""
    
def get_rot(pose3d):
#이 함수는 힙, 왼쪽 다리, 오른쪽 다리의 3차원 자세(3D pose)를 사용하여 회전을 계산하고 반환합니다.
    ## guesses
    hip_left = 2
    hip_right = 3
    hip_up = 16
    
    knee_left = 1
    knee_right = 4
    
    ankle_left = 0
    ankle_right = 5
    
    # hip
    
    x = pose3d[hip_right] - pose3d[hip_left]
    w = pose3d[hip_up] - pose3d[hip_left]
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    hip_rot = np.vstack((x, y, z)).T

    # right leg
    
    y = pose3d[knee_right] - pose3d[ankle_right]
    w = pose3d[hip_right] - pose3d[ankle_right]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_left] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_r_rot = np.vstack((x, y, z)).T

    # left leg
    
    y = pose3d[knee_left] - pose3d[ankle_left]
    w = pose3d[hip_left] - pose3d[ankle_left]
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = pose3d[hip_right] - pose3d[ankle_left]
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_l_rot = np.vstack((x, y, z)).T

    rot_hip = R.from_matrix(hip_rot).as_quat()
    rot_leg_r = R.from_matrix(leg_r_rot).as_quat()
    rot_leg_l = R.from_matrix(leg_l_rot).as_quat()
    
    return rot_hip, rot_leg_l, rot_leg_r
"""
def get_rot(pose3d)` 함수:
   이 함수는 힙, 왼쪽 다리, 오른쪽 다리의 3차원 자세(3D pose)를 사용하여 회전을 계산하고 반환합니다.

입력 3D 자세의 힙, 무릎, 발목 좌표를 추출하고, 이전에 언급한 `get_rot_hands(pose3d)` 및 `get_rot_mediapipe(pose3d)` 함수와 유사한 과정으로 회전 행렬을 계산한 다음 
이를 Quaternion 형태로 변환하고 반환합니다.

"""

def sendToPipe(text):
    if platform.startswith('win32'):
        pipe = open(r'\\.\pipe\ApriltagPipeIn', 'rb+', buffering=0)
        some_data = str.encode(text)
        some_data += b'\0'
        pipe.write(some_data)
        resp = pipe.read(1024)
        pipe.close()
    elif platform.startswith('linux'):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.connect("/tmp/ApriltagPipeIn")
        some_data = text.encode('utf-8')
        some_data += b'\0'
        client.send(some_data)
        resp = client.recv(1024)
        client.close()
    else:
        print(f"Unsuported platform {sys.platform}")
        raise Exception
    return resp
"""
`def sendToPipe(text)` 함수:
   이 함수는 텍스트를 입력 받아, 지정된 파이프(Windows 또는 Unix)를 사용하여하여 해당 텍스트를 전송하고 파이프에서 응답을 읽어 반환합니다.

Windows와 리눅스에 맞게 파일 및 소켓을 사용하여 파이프에 연결하고 텍스트를 전송한 다음 응답을 반환합니다.

"""

def sendToSteamVR_(text):
    #Function to send a string to my steamvr driver through a named pipe.
    #open pipe -> send string -> read string -> close pipe
    #sometimes, something along that pipeline fails for no reason, which is why the try catch is needed.
    #returns an array containing the values returned by the driver.
    try:
        resp = sendToPipe(text)
    except:
        return ["error"]

    string = resp.decode("utf-8")
    array = string.split(" ")
    
    return array
"""
`def sendToSteamVR_(text)` 함수:
  이 함수는 텍스트를 입력 받아, `sendToPipe(text)`을 호출하여 SteamVR 드라이버를 통해 텍스트를 전송하고 드라이버로부터 반환된 값을 배열로 반환합니다.

함수에서 발생한 예외를 처리하고 결과를 디코딩하여 문자열 값을 분할하고 이를 배열로 반환합니다.

"""

def sendToSteamVR(text, num_tries=10, wait_time=0.1):
    # wrapped function sendToSteamVR that detects failed connections
    ret = sendToSteamVR_(text)
    i = 0
    while "error" in ret:
        print("INFO: Error while connecting to SteamVR. Retrying...")
        time.sleep(wait_time)
        ret = sendToSteamVR_(text)
        i += 1
        if i >= num_tries:
            return None # probably better to throw error here and exit the program (assert?)
    
    return ret
"""
def sendToSteamVR(text, num_tries=10, wait_time=0.1)` 함수:
   이 함수는 `sendToSteamVR_(text)` 함수를 래핑하여 실패한 연결을 감지합니다.

`sendToSteamVR_(text)`를 호출하고 오류가 있는 경우 재시도하며 그 오류가 계속되는 경우 일정한 횟수에 도달한 후 반환 설정을 결정합니다 (예: None).
"""
    
class CameraStream():
    #이 클래스는 카메라 스트리밍을 관리하고 이미지를 지속적으로 캡처하는 역할을 합니다. 주요 기능은 다음과 같습니다:
    def __init__(self, params):
        self.params = params
        self.image_ready = False
        # setup camera capture
        if len(params.cameraid) <= 2:
            cameraid = int(params.cameraid)
        else:
            cameraid = params.cameraid
            
        if params.camera_settings: # use advanced settings
            self.cap = cv2.VideoCapture(cameraid, cv2.CAP_DSHOW) 
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        else:
            self.cap = cv2.VideoCapture(cameraid)  

        if not self.cap.isOpened():
            print("ERROR: Could not open camera, try another id/IP")
            shutdown(params)

        if params.camera_height != 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(params.camera_height))
            
        if params.camera_width != 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(params.camera_width))

        print("INFO: Start camera thread")
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()


    
    def update(self):
        # continuously grab images
        while True:
            ret, self.image_from_thread = self.cap.read()    
            self.image_ready = True
            
            if ret == 0:
                print("ERROR: Camera capture failed! missed frames.")
                self.params.exit_ready = True
                return
 
"""
`__init__(self, params)` 함수는 다음 인자들을 받습니다.

- `params`: 카메라에 대한 설정 값들을 담은 딕셔너리입니다. 카메라 아이디, 해상도, 프레임 레이트 등과 같은 정보가 들어있습니다.

`__init__` 함수내 주요 변수들은 다음과 같습니다.

- `self.camera_id`: 카메라의 고유 아이디입니다. 보통 0번 카메라는 기본 내장 카메라를 가리킵니다.
- `self.width`와 `self.height`: 영상의 가로, 세로 길이입니다.
- `self.capture`: `cv2.VideoCapture()`를 이용해 카메라와의 연결을 관리하는 객체입니다.
- `self.thread`: `update()` 함수를 비동기로 실행하기 위한 스레드입니다.

`update(self)` 함수는 카메라에서 실시간 영상을 받아와 처리하는 역할을 합니다.

- `ret, frame = self.capture.read()`: 카메라에서 현재 프레임을 읽어옵니다. 
`ret`는 읽기 성공 여부를 나타내며, `frame`은 받아온 영상입니다.
- `self.current_frame = frame`: 받아온 영상을 인스턴스 변수 `current_frame`에 저장합니다. 
이를 통해 다른 모듈에서 이 클래스를 사용하면 실시간 영상에 쉽게 접근할 수 있습니다.

`shutdown()` 함수는 객체가 소멸되기 전에 실행되어 카메라 연결을 종료하고 스레드를 정리합니다.

이 클래스를 사용하는 예시는 다음과 같습니다.

```python
params = {'camera_id': 0, 'width': 640, 'height': 480}
camera_stream = CameraStream(params)
# 실시간 영상 처리 작업
...
camera_stream.shutdown()
```

이 클래스는 다양한 카메라 기반 애플리케이션에서 사용될 수 있습니다. 
예를 들어, 얼굴 인식, 객체 감지, 동작 인식 등의 영상 처리를 실시간으로 수행하는 애플리케이션에 유용하게 사용됩니다.

"""
def shutdown(params):
    # first save parameters 
    print("INFO: Saving parameters...")
    params.save_params()

    cv2.destroyAllWindows()
    sys.exit("INFO: Exiting... You can close the window after 10 seconds.")
"""

`shutdown` 함수는 프로그램을 종료하기 전에 필요한 작업을 수행하는 함수입니다. 주요 목적은 파라미터를 저장하고 생성된 창을 닫는 것입니다.

```python
def shutdown(params):
    # 첫 번째로 파라미터를 저장합니다.
    print("INFO: Saving parameters...")
    params.save_params()
```
위 코드는 파라미터를 저장하는 부분입니다. `params`라는 객체에 있는 `save_params()` 함수를 호출하여 프로그램을 종료하기 전에 현재 파라미터를 저장합니다.
  
```python
    cv2.destroyAllWindows()
```
이 줄은 `cv2` 라이브러리에 생성된 창을 모두 닫는 함수입니다. 프로그램을 종료하기 전 이 함수를 호출하여 창을 닫으면 사용자 화면이 정리됩니다.
  
```python
    sys.exit("INFO: Exiting... You can close the window after 10 seconds.")
```
마지막으로, `sys.exit()` 함수를 호출하여 프로그램을 종료합니다. 인자로 전달된 메시지는 프로그램이 종료된 후 출력되는 메시지입니다.

요약하면, `shutdown(params)` 함수는 프로그램을 종료하기 전에 파라미터를 저장하고 생성된 창을 닫으며, 프로그램을 정상적으로 종료하여 사용자에게 종료 메시지를 전달하는 역할을 합니다.

"""