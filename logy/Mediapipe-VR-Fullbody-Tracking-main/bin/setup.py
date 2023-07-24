from cx_Freeze import setup, Executable
 
buildOptions = dict(packages=[], excludes = [])
 
exe = [Executable('C:\\logy\\Mediapipe-VR-Fullbody-Tracking-main\\bin\\mediapipepose.py')]
 
setup(
    name='logyApp',
    version='0.0.1',
    author='khw',
    description = 'description',
    package_data={'': ['templates/logy.png']},
    options = dict(build_exe = buildOptions),
    executables = exe
)