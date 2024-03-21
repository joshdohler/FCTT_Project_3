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

R0_average = 0.019554687500000018

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

np.savetxt('data.csv', (R1_list,C1_list), delimiter=',')

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


I_data = np.array(I_fit_R1[i:i2])

## For the fitting of R1

b1 = 0
def R1_final(I, r1, c1, r2, b2, c2):
    res = 0
    return r1*np.exp(-(I-b1)**2/c1)+r2*np.exp(-(I-b2)**2/c2)+0.0055

R1_data = np.array(R1_list[i:i2])

popt, popc = sci.curve_fit(R1_final, I_data, R1_data, maxfev = 999999999)

r1_opt1, c1_opt1, r2_opt2, b2_opt, c2_opt2 = popt
print(popt)
I_data.sort()

I_range = np.linspace(I_data[0], I_data[-1], 1000)

R1_fit = R1_final(I_range, r1_opt1, c1_opt1, r2_opt2, b2_opt, c2_opt2)



plt.figure(1)
plt.plot(I_fit_R1[i:i2],R1_list[i:i2],"ob")
plt.plot(I_range,R1_fit,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Current (A)')
plt.ylabel('Resistance R1 (Ohm)')
plt.legend(['Training data','Model'])
plt.show()