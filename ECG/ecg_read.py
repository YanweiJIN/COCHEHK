import numpy as np
import matplotlib.pyplot as plt

PATH="/Users/jinyanwei/Desktop/"     #path, 这里就是写刚才你保存的数据地址  
HEADERFILE="100.hea"     #文件格式为文本格式 
ATRFILE="100.atr"     #attributes-file 文件以二进制格式 
DATAFILE="100.dat"     #data-file  
SAMPLES2READ=3000     #读取的数据样本点数
####################读取头文件######################
f=open(PATH+HEADERFILE,"r")
z=f.readline().split()
nosig,sfreq=int(z[1]),int(z[2])     #% number of signals，sample rate of data

dformat,gain,bitres,zerovalue,firstvalue=[],[],[],[],[]
for i in range(nosig):
    z=f.readline().split()
    dformat.append(int(z[1]))     #format; here only 212 is allowed
    gain.append(int(z[2]))     #number of integers per mV
    bitres.append(int(z[3]))     #bitresolution
    zerovalue.append(int(z[4]))     #integer value of ECG zero point
    firstvalue.append(int(z[5]))     #first integer value of signal (to test for errors)
f.close()
####################读取dat文件######################
f=open(PATH+DATAFILE,"rb")     #以二进制格式读入dat文件
b=f.read()   
f.close()

A_init=np.frombuffer(b,dtype=np.uint8)      #将读入的二进制文件转化为unit8格式
A_shape0=int(A_init.shape[0]/3)     
A=A_init.reshape(A_shape0,3)[:SAMPLES2READ]     #将A转为3列矩阵

M=np.zeros((SAMPLES2READ,2))     #创建矩阵M

M2H=A[:,1]>>4     #字节向右移四位，即取字节的高四位
M1H=A[:,1]&15     #取字节的低四位

PRL=(A[:,1]&8)*(2**9)     #sign-bit   取出字节低四位中最高位，向左移九位，等于乘2^9
PRR=A[:,1]&128<<5     #sign-bit   取出字节高四位中最高位，向左移五位

M1H=M1H*(2**8)
M2H=M2H*(2**8)

M[:,0]=A[:,0]+M1H-PRL
M[:,1]=A[:,2]+M2H-PRR

if ((M[1,:]!=firstvalue).any()):
    print("inconsistency in the first bit values")

if nosig==2:
    M[:, 0] = (M[:, 0] - zerovalue[0]) / gain[0]
    M[:, 1] = (M[:, 1] - zerovalue[1]) / gain[1]
    TIME=np.linspace(0,SAMPLES2READ-1,SAMPLES2READ)/sfreq
elif nosig==1:
    M2=[]
    M[:, 0] = M[:, 0] - zerovalue[0]
    M[:, 1] = M[:, 1] - zerovalue[1]
    for i in range(M.shape[0]):
        M2.append(M[:,0][i])
        M2.append(M[:,1][i])
    M2.append(0)
    del M2[0]
    M2=np.array(M2)/gain[0]
    TIME=np.linspace(0,2*SAMPLES2READ-1,2*SAMPLES2READ)/sfreq
else:
    print("Sorting algorithm for more than 2 signals not programmed yet!")
####################读取atr文件######################
f=open(PATH+ATRFILE,"rb")     #主要是读取ATR文件中各周期数据并在之后打印在图中
b=f.read()
f.close()

A_init=np.frombuffer(b,dtype=np.uint8)
A_shape0=int(A_init.shape[0]/2)
A=A_init.reshape(A_shape0,2)

ANNOT,ATRTIME=[],[]
i=0
while i < A.shape[0]:
    annoth=A[i,1]>>2
    if annoth==59:
        ANNOT.append(A[i+3,1]>>2)
        ATRTIME.append(A[i+2,0]+A[i+2,1]*(2**8)+A[i+1,0]*(2**16)+A[i+1,1]*(2**24))
        i+=3
    elif annoth==60:pass
    elif annoth==61:pass
    elif annoth==62:pass
    elif annoth==63:
        hilfe=(A[i,1]&3)*(2**8)+A[i,0]
        hilfe=hilfe+hilfe%2
        i+=int(hilfe/2)
    else:
        ATRTIME.append((A[i,1]&3)*(2**8)+A[i,0])
        ANNOT.append(A[i,1]>>2)
    i+=1

del ANNOT[len(ANNOT)-1]
del ATRTIME[len(ATRTIME)-1]

ATRTIME=np.array(ATRTIME)
ATRTIME=np.cumsum(ATRTIME)/sfreq

ind=np.where(ATRTIME<=TIME[-1])[0]
ATRTIMED=ATRTIME[ind]

ANNOT=np.round(ANNOT)
ANNOTD=ANNOT[ind]
#####################显示ECG####################
plt.plot(TIME,M[:,0],linewidth="0.5",c="r")
if nosig==2:
    plt.plot(TIME, M[:, 1], linewidth="0.5", c="b")
for i in range(len(ATRTIMED)):
    plt.text(ATRTIMED[i],0,str(ANNOTD[i]))
plt.xlim(TIME[0],TIME[-1])
plt.xlabel("Time / s")
plt.ylabel("Votage / mV")
plt.title("ECG signal ")
plt.show()




################
import wfdb
record = wfdb.rdsamp('/Users/jinyanwei/Desktop/100', sampto=3000)
annotation = wfdb.rdann('/Users/jinyanwei/Desktop/100', 'atr', sampto=3000)
print("Annotation file information:")
print("Record name:", annotation.record_name)
print("Extension:", annotation.extension)
#print("Sample types:", annotation.sample_type)
print("Sample locations:", annotation.sample)
print("Symbol types:", annotation.symbol)
print("Auxiliary symbols:", annotation.aux_note)
index = 10
print("\nAnnotation at index", index)
print("Sample:", annotation.sample[index])
print("Symbol:", annotation.symbol[index])
print("Auxiliary note:", annotation.aux_note[index])

import pandas as pd
data = {
    'sample': annotation.sample,
    'symbol': annotation.symbol,
    'label_store': annotation.aux_note,
}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)



############################
if os.path.isdir("mitdb"):
    print('You already have the data.')
else:
    wfdb.dl_database('mitdb', 'mitdb')




#############################
import numpy as np

# Path where data is saved
PATH = '/Users/jinyanwei/Desktop/BP_Model/Jinyw_code/mitdb'

# Attributes file in binary format
ATRFILE = '100.atr'
atrd = os.path.join(PATH, ATRFILE)

# Read attribute file with annotation data
with open(atrd, 'rb') as fid3:
    A = np.fromfile(fid3, dtype=np.uint8).reshape(-1, 2)
sfreq = A[1, 1]

SAMPLES2READ = 1
ATRTIME = []
ANNOT = []
TIME = np.arange(SAMPLES2READ) / sfreq

saa = A.shape[0]
i = 0
while i < saa:
    annoth = A[i, 1] >> 2
    if annoth == 59:
        ANNOT.append(A[i + 3, 1] >> 2)
        ATRTIME.append(
            A[i + 2, 0] + (A[i + 2, 1] << 8) + (A[i + 1, 0] << 16) + (A[i + 1, 1] << 24)
        )
        i += 3
    elif annoth == 60:
        pass
    elif annoth == 61:
        pass
    elif annoth == 62:
        pass
    elif annoth == 63:
        hilfe = (A[i, 1] & 3) << 8 | A[i, 0]
        hilfe += hilfe % 2
        i += int(hilfe / 2)
    else:
        ATRTIME.append((A[i, 1] & 3) << 8 | A[i, 0])
        ANNOT.append(A[i, 1] >> 2)
    i += 1

ANNOT.pop()  # Last line = EOF (=0)
ATRTIME.pop()  # Last line = EOF

ATRTIME = np.cumsum(ATRTIME) / sfreq
ind = np.where(ATRTIME <= TIME[-1])
ATRTIMED = ATRTIME[ind]
ANNOT = np.round(ANNOT)



########################
print("Annotation file information:")
print("Record name:", annotation.record_name)
print("Extension:", annotation.extension)
#print("Sample types:", annotation.sample_type)
print("Sample locations:", annotation.sample)
print("Symbol types:", annotation.symbol)
print("Auxiliary symbols:", annotation.aux_note)
index = 10
print("\nAnnotation at index", index)
print("Sample:", annotation.sample[index])
print("Symbol:", annotation.symbol[index])
print("Auxiliary note:", annotation.aux_note[index])

import pandas as pd
data = {
    'sample': annotation.sample,
    'symbol': annotation.symbol,
    'label_store': annotation.aux_note,
}

# Create a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)