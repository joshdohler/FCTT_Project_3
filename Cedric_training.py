import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Q = 2.5 #rate discharge capacity [A.h]

Model_Training = pd.read_csv('Model_Training_Data_20.csv')

Time = np.array(Model_Training['Time (s)'])

N = len(Time)

V_model = np.array(Model_Training['Voltage (V)'])

I_model = np.array(Model_Training['Current (A)'])

time_end_impulse = []

R0_list = []
R1_list = []
DeltaT_list = []
C1_list = []

for i in range (N):
    if I_model[i]!=0 and I_model[i+1]==0:
        time_end_impulse.append(Time[i])
        dV0 = V_model[i+1]-V_model[i]
        dI0 = I_model[i+1]-I_model[i]
        R0 = dV0/dI0
        R0_list.append(R0)
        dVinf = V_model[i+3000]-V_model[i]
        dIinf = I_model[i+3000]-I_model[i]
        R1 = dVinf/dIinf-R0
        R1_list.append(R1)
        j = i
        V_converged = V_model[j]
        while V_converged != V_model[i+3000]:
            j +=1
            V_converged = V_model[j]
        DeltaT = Time[j]-Time[i]
        DeltaT_list.append(DeltaT)
        C1 = DeltaT/(4*R1)
        C1_list.append(C1)


plt.figure(1)
plt.plot(Time,V_model)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(['Experimental'])
plt.show()

plt.figure(2)
plt.plot(Time,I_model)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend(['Experimental'])
plt.show()