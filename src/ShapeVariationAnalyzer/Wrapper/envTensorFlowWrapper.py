#!/usr/bin/env python

import os, sys
import subprocess
import json
import ast


def wrap(cmd_setenv, program, args=0):
    bashCommand = cmd_setenv + " " + '\"' + program + '\"'
    for flag,value in args.items():
        bashCommand = bashCommand + " " + flag + ' \"' + value + '\"'

    command = ["bash", "-c", str(bashCommand)]

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err =  p.communicate()
    print("\nout : " + str(out) + "\nerr : " + str(err))

def config_env():
    # 
    # ----- Virtualenv setup ----- #
    # 
    """ Virtualenv setup with Tensorflow
    1 - install virtualenv if it's not already
    2 - create a virtualenv into slicer temporary path
    3 - install tensorflow into the virtualenv
    """ 
    pathSlicerExec = str(os.path.dirname(sys.executable))
    if sys.platform == 'win32':
        pathSlicerExec.replace("/","\\")
    currentPath = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == 'win32':
        Python35 = 0
        for path in os.getenv('PATH').split(';'):
            if "Python35" in path and "Scripts" in path:
                    pathSlicerPython = '\"' + os.path.join(path, '..', "python.exe") + '\"'
                    dirSitePckgs = os.path.join(path, "..", "Lib", "site-packages")
                    pythonpath = '\"' + os.path.join(path, '..') + '\"'
                    Python35 = 1
                    break
        if Python35 == 0:
            # install python 3.5 + virtualenv
            # @powershell -ExecutionPolicy Unrestricted "iex ((new-object net.webclient).DownloadString('https://raw.githubusercontent.com/ClementMirabel/ShapeVariationAnalyzer-Resources/master/install-SVA-deps.ps1'))"
            powershellPath = os.path.join(currentPath, "..", "Resources", "PowerShell", "install-SVA-deps.ps1")

            p = subprocess.Popen(['powershell.exe', "Set-ExecutionPolicy RemoteSigned"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err =  p.communicate()
            # print("out : " + str(out) + "\nerr : " + str(err))

            p = subprocess.Popen(['powershell.exe', powershellPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err =  p.communicate()
            print("out : " + str(out) + "\nerr : " + str(err))

            print "\n\nTERMINEY\n\n"
            rootDir = os.path.splitdrive(sys.executable)[0] + '/'
            for root in os.listdir(rootDir):
                if "Python35" in root:
                    pythonpath = '\"' + os.path.join(rootDir, root) + '\"'
                    print pythonpath
                    pathSlicerPython = '\"' + os.path.join(rootDir, root, "python.exe") + '\"'
                    print pathSlicerPython
                    dirSitePckgs = os.path.join(rootDir, root, "Lib", "site-packages")
                    print dirSitePckgs
                    os.environ["path"] = os.getenv("path") + os.path.join(rootDir, root) + ";" + os.path.join(rootDir, root, 'Scripts') + ";" 
                    break

    else: 
        dirSitePckgs = os.path.join(pathSlicerExec, "..", "lib", "Python", "lib",'python%s' % sys.version[:3], "site-packages")
        pathSlicerPython = os.path.join(pathSlicerExec, "..", "bin", "SlicerPython")
    import pip
    # Check virtualenv installation
    print("\n\n I. Virtualenv installation")
    try:
        import virtualenv
        print("===> Virtualenv already installed")
    except Exception as e: 
        venv_install = pip.main(['install', 'virtualenv'])
        import virtualenv
        print("===> Virtualenv now installed with pip.main")


    print("\n\n II. Create environment tensorflowSlicer")
    currentPath = os.path.dirname(os.path.abspath(__file__))
    tempPath = os.path.join(currentPath, '..', 'Resources')
    env_dir = os.path.join(tempPath, "env-tensorflow") 
    if not os.path.isdir(env_dir):
        os.mkdir(env_dir) 

    if not os.path.isfile(os.path.join(env_dir, 'bin', 'activate')):
        if sys.platform == 'win32': 
            command = ["bash", "-c", 'export PYTHONPATH=' + pythonpath + '; export PYTHONHOME=' + pythonpath + '; ' + pathSlicerPython + ' \"' + os.path.join(dirSitePckgs, 'virtualenv.py') + '\" --python=' + pathSlicerPython + ' \"' + env_dir + '\"']
        else:
            command = ["bash", "-c", pathSlicerPython + ' \"' + os.path.join(dirSitePckgs, 'virtualenv.py') + '\" --python=' + pathSlicerPython + ' \"' + env_dir + '\"']
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err =  p.communicate()
        print("out : " + str(out) + "\nerr : " + str(err))
        print("\n===> Environmnent tensorflowSlicer created")


    print("\n\n\n III. Install tensorflow into tensorflowSlicer")
    """ To install tensorflow in virtualenv, requires:
        - activate environment
        - export PYTHONPATH
        - launch python
            => cmd_setenv
        - add the environment path to sys.path
        - set sys.prefix 
        - pip install tensorflow
      """
    # source path-to-env/bin/activate
    if sys.platform == 'win32': 
        cmd_setenv = '\"' + os.path.join(env_dir, 'Scripts', 'activate') + '\"; '
    else:
        cmd_setenv = "source " + os.path.join(env_dir, 'bin', 'activate') + "; "
    # construct python path
    if sys.platform == 'win32': 
        env_pythonpath = '\"' + os.path.join(env_dir, 'Scripts') + ';' + os.path.join(env_dir, 'Lib') + ';' + os.path.join(env_dir, 'Lib', 'site-packages') + '\"'
        env_pythonhome = '\"' + os.path.join(env_dir, 'Scripts') + '\"'
    else:
        env_pythonpath = os.path.join(env_dir, 'bin') + ":" + os.path.join(env_dir, 'lib', 'python%s' % sys.version[:3]) + ":" + os.path.join(env_dir, 'lib', 'python%s' % sys.version[:3], 'site-packages')
        env_pythonhome = os.path.join(env_dir, 'bin')
    # export python path
    cmd_setenv = cmd_setenv + "export PYTHONPATH=" + env_pythonpath +  "; " + "export PYTHONHOME=" + env_pythonhome +  "; "
    # call Slicer python
    cmd_setenv = cmd_setenv + pathSlicerPython

    # construct sys.path
    if sys.platform == 'win32': 
        # env_syspath = "sys.path.append(\"" + os.path.join(env_dir,'Lib') + "\"); sys.path.append(\"" + os.path.join(env_dir, 'Lib', 'site-packages') + "\"); sys.path.append(\"" + os.path.join(env_dir,'Lib','site-packages','pip','utils') + "\"); "
        env_syspath = "sys.path.append(\"" + os.path.join(env_dir,'Lib') + "\"); sys.path.append(\"" + os.path.join(env_dir, 'Lib', 'site-packages') + "\"); "
    else:
        env_syspath = "sys.path.append(\"" + os.path.join(env_dir,'lib', 'python%s' % sys.version[:3]) + "\"); sys.path.append(\"" + os.path.join(env_dir,'lib','python%s' % sys.version[:3], 'site-packages') + "\"); sys.path.append(\"" + os.path.join(env_dir,'lib','python%s' % sys.version[:3], 'site-packages','pip','utils') + "\"); "
    cmd_virtenv = str(' -c ')
    cmd_virtenv = cmd_virtenv + "\'import sys; " + env_syspath 

    # construct sys.path
    env_sysprefix = "sys.prefix=\"" + env_dir + "\"; "
    cmd_virtenv = cmd_virtenv + env_sysprefix

    # construct install command
    env_install = "import pip; pip.main([\"install\", \"--prefix=" + env_dir + "\", \"tensorflow\"]); pip.main([\"install\", \"--prefix=" + env_dir + "\", \"pandas\"])\'"
    cmd_virtenv = cmd_virtenv + env_install

    bashCommand = cmd_setenv + cmd_virtenv
    command = ["bash", "-c", str(bashCommand)]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err =  p.communicate()
    print("\nout : " + str(out) + "\nerr : " + str(err))

    # Tensorflow is now installed but might not work due to a missing file
    # We create it to avoid the error 'no module named google.protobuf'
    # -----
    print("\n\n Create missing __init__.py if doesn't existe yet")
    if sys.platform == 'win32': 
        google_init = os.path.join(env_dir, 'Lib', 'site-packages', 'google', '__init__.py')
    else:
        google_init = os.path.join(env_dir, 'lib', 'python%s' % sys.version[:3], 'site-packages', 'google', '__init__.py')
    if not os.path.isfile(google_init):
        file = open(google_init, "w")
        file.close()
    

    print("\n\n\n IV. Check tensorflow is well installed")
    test_tf = os.path.join(currentPath, "..","Testing", "test-tensorflowinstall.py")
    bashCommand = cmd_setenv + " " + test_tf

    command = ["bash", "-c", bashCommand]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err =  p.communicate()
    print("\nout : " + str(out) + "\nerr : " + str(err))

    return cmd_setenv

def main():
    cmd_setenv = config_env()

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-pgm', action='store', dest='program')
    parser.add_argument('-args', action='store', dest='args')

    args = parser.parse_args()
    wrap(cmd_setenv, args.program, ast.literal_eval(args.args))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)


