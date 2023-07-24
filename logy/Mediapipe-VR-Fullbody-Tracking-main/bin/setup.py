from cx_Freeze import setup, Executable
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
 
exe = [Executable('C:\\logy\\Mediapipe-VR-Fullbody-Tracking-main\\bin\\mediapipepose.py')]
 
options = {
    'build_exe': {
        'include_files': [
             (os.path.join(current_dir, 'templates'), 'templates'),
        ],
    },
}

setup(
    name='logyApp',
    version='0.0.1',
    author='khw',
    description = 'description',
    options=options,
    executables=[Executable(os.path.join(current_dir, "mediapipepose.py"))]
)