import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Q = 2.5 #rate discharge capacity [A.h]

SOC_OCV = pd.read_csv('SOC_OCV_MFCTT_2019.csv')

SOC = SOC_OCV['SOC'][:]
SOC_step = SOC[0]-SOC[1]

OCV = SOC_OCV['Ecell/V'][:]

Model_Training = pd.read_csv('Battery_Testing_Data.csv')

Time = list(np.array(Model_Training['Time (s)'][18000:]))

N = len(Time)

V_model_training = np.array(Model_Training['Voltage (V)'][18000:])

I_model_training = np.array(Model_Training['Current (mA)'][18000:])*10**(-3)


## My RC model

R0_average = 0.019554687500000018

C1_average = 1962.9469051907095

R101 = 1.48113763e-01
R102 = 5.65859345e-03
b1 = 0
b2 = 1.26842729e+00
c1 = 3.30006677e-01
c2 = 1.19474291e+02
Offset = 0.0055

def R1_final(I):
    res = 0
    return R101*np.exp(-(I-b1)**2/c1)+R102*np.exp(-(I-b2)**2/c2)+Offset

z = [0 for i in range(N)]

z[0] = 90

V_model = [0 for i in range(N)]

I_model = -I_model_training

I_R1 = [0 for i in range(N)]

I_R1[0] = 0      #is this right?

R1_final_list = []

OCV_cont = [0 for i in range(N)]

R1_cont = [0 for i in range(N)]

R0_cont = [0 for i in range(N)]

R1_actual = R1_final(I_model[0])

for k in range(N):
    SOCindex = round((SOC[0]-z[k])/SOC_step)
    OCV[SOCindex]
    OCV_cont[k] = OCV[SOCindex]
    if I_model[k]!=0:
        R1_actual = R1_final(I_model[k])
    C1_actual = C1_average
    R0_actual = R0_average
    R1_cont[k] = R1_actual*I_R1[k]
    R0_cont[k] = R0_actual*I_model[k]
    V_model[k] = OCV[SOCindex]-R1_actual*I_R1[k]-R0_actual*I_model[k]
    R1_final_list.append(R1_actual)
    if k!=N-1:
        DeltaT = Time[k+1]-Time[k]
        z[k+1] = z[k]-I_model[k]*(DeltaT)*100/(Q*3600)
        I_R1[k+1] = np.exp(-DeltaT/(R1_actual*C1_actual))*I_R1[k]+(1-np.exp(-DeltaT/(R1_actual*C1_actual)))*I_model[k]


## Plots

plt.figure(1)
plt.plot(Time,V_model_training)
plt.plot(Time,V_model,color='red', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,R0_cont,color='black', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,R1_cont,color='green', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,OCV_cont,color='purple', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(['Testing data','Model'])
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
error = (V_model-V_model_training)/V_model_training*100
ax1.plot(Time,error,'b-')
ax2.plot(Time,I_model,'g-')
#plt.plot(Time,R1_cont,color='green', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,I_R1,color='red', linestyle='dashed', linewidth=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Percentage Error - Voltage (%)', color='b')
ax1.set_ylim([-4,4])
ax2.set_ylabel('Current (A)', color='g')
ax2.set_ylim([-15.5,15.5])
plt.show()

plt.figure(3)
plt.plot(Time,error,'b-')
plt.xlabel('Time (s)')
plt.ylabel('Percentage Error - Voltage (%)', color='b')
plt.show()

Error_total = 0
for k in range(N):
    Error_total += abs(V_model_training[k]-V_model[k])

Error_total = Error_total/N
print('Error_total=',Error_total)


plt.figure(4)
plt.plot(Time,z)
plt.xlabel('Time (s)')
plt.ylabel('SOC (%)')
plt.legend(['Model'])
plt.show()

plt.figure(5)
plt.plot(Time,R1_final_list)
plt.xlabel('Time (s)')
plt.ylabel('R1_final (Ohm)')
plt.legend(['Model'])
plt.show()