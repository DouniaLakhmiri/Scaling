import os
import sys


if len(sys.argv) != 2:
    print('Usage of pytorch_bb.py: X.txt')
    exit()


fin = open(sys.argv[1], 'r')
Lin = fin.readlines()
Xin = Lin[0].split()
fin.close()

syst_cmd = 'python3 blackbox.py '

for i in range(len(Xin)):
    syst_cmd += str(Xin[i]) + ' '

syst_cmd += '> out.txt 2>&1'
os.system(syst_cmd)

fout = open('out.txt', 'r')
Lout = fout.readlines()
for line in Lout:
    if "Final accuracy" in line:
        tmp = line.split()
        acc = '-' + str(tmp[3])
        fout.close()

    if 'MACS and FLOPS' in line:
        tmp = line.split()
        flops_ratio = str(tmp[-1])
        macs_ration = str(tmp[-2])

        print(acc, macs_ration, flops_ratio)
        exit()

print('Inf', 'Inf', 'Inf')
fout.close()
