import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sci

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

Time_change_SOC = [8939.679, 30036.056, 51132.209, 72228.311, 93324.521, 114421.016, 135517.358, 156613.506];

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

time_end_impulse_R1 = []
time_converged_R1 = []
V_impulse_R1 = []
V_converged_list_R1 = []

R0_average = 0.019554687500000018

for i in range (N):
    if I_model_training[i]!=0 and I_model_training[i+1]==0 and Time[i] not in Time_change_SOC:
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
    elif I_model_training[i]!=0 and I_model_training[i+1]==0 and Time[i]==72228.311:
        dVinf = abs(V_model_training[i+3000]-V_model_training[i])
        dIinf = abs(I_model_training[i+3000]-I_model_training[i])
        V_impulse_R1.append(V_model_training[i])
        V_converged_list_R1.append(V_model_training[i+3000])
        time_end_impulse_R1.append(Time[i])
        j = i
        V_converged = V_model_training[j]
        while V_converged != V_model_training[i+3000]:
            j +=1
            V_converged = V_model_training[j]
        time_converged_R1.append(Time[j])
        R1_discharge_60 = dVinf/dIinf-R0_average
        I_discharge_60 = I_model_training[i]


## My RC model

R0_average = sum(R0_list)/len(R0_list)

C1_average = sum(C1_list)/len(C1_list)

R1_average = sum(R1_list)/len(R1_list)


## For the plot of R1 @60% SOC

z = [0 for i in range(N)]

z[0] = 100

for k in range(N-1):
    DeltaT = Time[k+1]-Time[k]
    z[k+1] = z[k]+I_model_training[k]*(DeltaT)*100/(Q*3600)

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

I_data = I_fit_R1[i:i2]
I_data.append(I_discharge_60)
I_data = np.array(I_data)

## For the fitting of R1

b1 = 0

def R1_final(I, r1, c1, r2, b2, c2):
    return r1*np.exp(-(I-b1)**2/c1)+r2*np.exp(-(I-b2)**2/c2)+0.0055

R1_data = R1_list[i:i2]
R1_data.append(R1_discharge_60)
R1_data = np.array(R1_data)

popt, popc = sci.curve_fit(R1_final, I_data, R1_data, maxfev = 999999999)

r1_opt1, c1_opt1, r2_opt2, b2_opt2, c2_opt2 = popt


I_range = np.linspace(min(I_data), max(I_data), 1000)

R1_fit = R1_final(I_range, r1_opt1, c1_opt1, r2_opt2, b2_opt2, c2_opt2)



plt.figure(1)
plt.plot(I_data,R1_data,"ob")
plt.plot(I_range,R1_fit,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Current (A)')
plt.ylabel('$R_1$ ($\Omega$)')
plt.legend(['Training data','Model'])
plt.show()

plt.figure(2)
plt.plot(Time,z)
plt.xlabel('Time (s)')
plt.ylabel('SOC (%)')
plt.legend(['Model'])
plt.show()

plt.figure(3)
plt.plot(I_fit_R1[i:i2],R0_list[i:i2],"ob")
plt.xlabel('Current (A)')
plt.ylabel('$R_0$ ($\Omega$)')
plt.show()

plt.figure(4)
plt.plot(I_fit_R1[i:i2],C1_list[i:i2],"ob")
plt.xlabel('Current (A)')
plt.ylabel('$C_1$ ($F$)')
plt.show()

plt.figure(5)
plt.plot(time_converged,V_converged_list,"ob")
plt.plot(time_end_impulse,V_impulse,"or")
plt.plot(time_end_impulse_plus1,V_impulse_plus1,"og")
plt.plot(Time,V_model_training)
plt.xlabel('Time (s)')
plt.ylabel('Potential (V)')
plt.legend(['$V_{converged}$','$V_{end-impulse}$','$V_{end-impulse+1}$','$V_{training}$'])
plt.show()

for k in range(i,i2):
    time_converged_R1.append(time_converged[k])
    V_converged_list_R1.append(V_converged_list[k])
    time_end_impulse_R1.append(time_end_impulse[k])
    V_impulse_R1.append(V_impulse[k])

plt.figure(6)
plt.plot(time_converged_R1,V_converged_list_R1,"ob")
plt.plot(time_end_impulse_R1,V_impulse_R1,"or")
plt.plot(Time,V_model_training)
plt.xlabel('Time (s)')
plt.ylabel('Potential (V)')
plt.legend(['$V_{converged}$','$V_{end-impulse}$','$V_{training}$'])
plt.show()

