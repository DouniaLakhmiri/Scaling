import os
import sys


if len(sys.argv) != 2:
    print ('Usage of pytorch_bb.py: X.txt')
    exit()


fin = open(sys.argv[1], 'r')
Lin = fin.readlines()
Xin = Lin[0].split()
fin.close()

syst_cmd = 'python3 blackbox2.py '

for i in range(len(Xin)):
    syst_cmd += str(Xin[i]) + ' '

syst_cmd += '> out.txt 2>&1'
os.system(syst_cmd)

fout = open('out.txt', 'r')
Lout = fout.readlines()
for line in Lout:
    
    if 'MACS' in line:
        tmp = line.split()
        macs = str(tmp[-1])
        
        
    if "Accuracy" in line:
        tmp = line.split()
        diff_acc = str(tmp[-1])
        
        
    if "Parameters" in line: 
        tmp = line.split()
        params = str(tmp[-1])
        fout.close()
    

        print(macs, diff_acc, params)
        exit()

print('Inf', 'Inf', 'Inf')
fout.close()

