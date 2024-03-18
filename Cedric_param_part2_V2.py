import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Q = 2.5 #rate discharge capacity [A.h]

SOC_OCV = pd.read_csv('SOC_OCV_MFCTT_2019.csv')

SOC = SOC_OCV['SOC'][:]
SOC_step = SOC[0]-SOC[1]

OCV = SOC_OCV['Ecell/V'][:]

Model_Training = pd.read_csv('Model_Training_Data_20.csv')

Time = list(np.array(Model_Training['Time (s)'][:]))

N = len(Time)

V_model_training = np.array(Model_Training['Voltage (V)'][:])

I_model_training = np.array(Model_Training['Current (A)'][:])

## Parametrisation

time_end_impulse = []
time_converged = []
time_end_impulse_plus1 = []
R0_list = []
R1_list = []
I_fit_R1 = []
DeltaT_list = []
C1_list = []
V_impulse = []
V_impulse_plus1 = []
V_converged_list=[]

R0_average = 0.019540297174958105

for i in range (N):
    if I_model_training[i]!=0 and I_model_training[i+1]==0:
        time_end_impulse.append(Time[i])
        time_end_impulse_plus1.append(Time[i+1])
        V_impulse.append(V_model_training[i])
        V_impulse_plus1.append(V_model_training[i+1])
        dV0 = V_model_training[i+1]-V_model_training[i]
        dI0 = I_model_training[i+1]-I_model_training[i]
        R0 = dV0/dI0
        R0_list.append(R0)
        dVinf = abs(V_model_training[i+3000]-V_model_training[i])
        dIinf = abs(I_model_training[i+3000]-I_model_training[i])
        R1 = dVinf/dIinf-R0_average
        R1_list.append(R1)
        I_fit_R1.append(I_model_training[i])
        j = i
        V_converged = V_model_training[j]
        while V_converged != V_model_training[i+3000]:
            j +=1
            V_converged = V_model_training[j]
        time_converged.append(Time[j])
        V_converged_list.append(V_converged)
        DeltaT = Time[j]-Time[i]
        DeltaT_list.append(DeltaT)
        C1 = DeltaT/(4*R1)
        C1_list.append(C1)

## My RC model

R0_average = sum(R0_list)/len(R0_list)

C1_average = sum(C1_list)/len(C1_list)

R1_average = sum(R1_list)/len(R1_list)

R101 = 0.07           #@0: 0.06, @20: 0.07, @40: 0.05
R102 = 0.006        #@0: 0.017, @20: 0.006, @40: 0.003
b1 = -0.5         #@0: -0.5, @20: -0.5, @40: -0.5
b2 = b1
c1 = 1          #@0: 0.5, @20: 1, @40: 1
c2 = 30           #@0: 70, @20: 30, @40: 60
Offset = 0.0055           #@0: 0.025, @20: 0.0055, @40: 0

def R1_final(I):
    res = 0
    return R101*np.exp(-(I-b1)**2/c1)+R102*np.exp(-(I-b2)**2/c2)+Offset

z = [0 for i in range(N)]

z[0] = 100

V_model = [0 for i in range(N)]

I_model = -I_model_training

I_R1 = [0 for i in range(N)]

I_R1[0] = 0      #is this right?

R1_final_list = []

OCV_cont = [0 for i in range(N)]

R1_cont = [0 for i in range(N)]

R0_cont = [0 for i in range(N)]

j=0

R1_actual = R1_final(I_model[0])

for k in range(N):
    SOCindex = round((SOC[0]-z[k])/SOC_step)
    OCV[SOCindex]
    OCV_cont[k] = OCV[SOCindex]
    if I_model[k]==0 and I_model[k-1]!=0:
        R1_actual = R1_final(I_model[k-1])
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


## Plot R1 @60SOC

i = 0
k = Time.index(time_end_impulse[i])
while abs(z[k]-60)>5:
    i+=1
    k = Time.index(time_end_impulse[i])

i2 = i
k2 = k
while abs(z[k2]-60)<5:
    i2+=1
    k2 = Time.index(time_end_impulse[i2])

abscisse = I_fit_R1[i:i2]
abscisse.sort()
abscisse = np.linspace(abscisse[0], abscisse[-1], 1000)

R1f_list = [R1_final(i) for i in abscisse]

plt.figure(1)
plt.plot(I_fit_R1[i:i2],R1_list[i:i2],"ob")
plt.plot(abscisse,R1f_list,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Current (A)')
plt.ylabel('Resistance R1 (Ohm)')
plt.legend(['Training data','Model'])
plt.show()

plt.figure(2)
plt.plot(Time,V_model_training)
plt.plot(Time,V_model,color='red', linestyle='dashed', linewidth=0.5)
plt.plot(Time,R0_cont,color='black', linestyle='dashed', linewidth=0.5)
plt.plot(Time,R1_cont,color='green', linestyle='dashed', linewidth=0.5)
plt.plot(Time,OCV_cont,color='purple', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(['Training data','Model','Contribution R0','Contribution R1','Contribution OCV'])
plt.show()

plt.figure(3)
plt.plot(Time,I_model)
plt.plot(Time,R1_cont,color='green', linestyle='dashed', linewidth=0.5)
plt.plot(Time,I_R1,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend(['Training data','Contribution R1','I_R1'])
plt.show()

plt.figure(4)
plt.plot(Time,z)
plt.xlabel('Time (s)')
plt.ylabel('SOC (%)')
plt.legend(['Model'])
plt.show()

plt.figure(5)
plt.plot(I_fit_R1[i:i2],R1_list[i:i2],"ob")
plt.xlabel('Current (A)')
plt.ylabel('Resistance R0 (Ohm)')
plt.legend(['Training data'])
plt.show()

plt.figure(6)
plt.plot(I_fit_R1[i:i2],C1_list[i:i2],"ob")
plt.xlabel('Current (A)')
plt.ylabel('Capacité (F)')
plt.legend(['Training data'])
plt.show()

plt.figure(7)
plt.plot(Time,R1_final_list)
plt.xlabel('Time (s)')
plt.ylabel('R1_final (Ohm)')
plt.legend(['Model'])
plt.show()

plt.figure(8)
plt.plot(time_converged,V_converged_list,"ob")
plt.plot(time_end_impulse,V_impulse,"or")
plt.plot(time_end_impulse_plus1,V_impulse_plus1,"og")
plt.plot(Time,V_model_training)
plt.xlabel('Time (s)')
plt.ylabel('Potential (V)')
plt.legend(['V_converged','V_end_impulse','V_end_impulse_plus1','V_model_training'])
plt.show()