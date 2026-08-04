import subprocess
import os

f = os.path.abspath('mineudec.py')
print('PATH utf8 =', f.encode('utf-8'))
r = subprocess.call([r'C:\py312\python.exe', '-c',
                     'import sys; print("ARG bytes=", sys.argv[1].encode("utf-8"))', f], cwd='.')
print('ret', r)