#!/usr/bin/env python

import os
import subprocess

os.system(r"rm     build/*' 2.'*")
os.system(r"rm   build/*/*' 2.'*")
os.system(r"rm build/*/*/*' 2.'*")

os.system(r"rm     build/*' 3.'*")
os.system(r"rm   build/*/*' 3.'*")
os.system(r"rm build/*/*/*' 3.'*")

os.system(r"rm     build/*' 4.'*")
os.system(r"rm   build/*/*' 4.'*")
os.system(r"rm build/*/*/*' 4.'*")

output = subprocess.check_output("mount", shell=True)
if "//sinisa@surfers.engr.ucr.edu/coh on /Volumes/coh (smbfs, nodev, nosuid, mounted by sinisa)" not in output.decode("utf8"):
    print(output)
    print("You need to mount sinisa@surfers.engr.ucr.edu/coh")
    exit()

os.system("rsync -av --progress --exclude=\".*\" build/html/ /Volumes/coh/wfbase/")
