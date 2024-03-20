import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Q = 2.5 #rate discharge capacity [A.h]

SOC_OCV = pd.read_csv('SOC_OCV_MFCTT_2019.csv')

SOC = SOC_OCV['SOC'][:]
SOC_step = SOC[0]-SOC[1]

OCV = SOC_OCV['Ecell/V'][:]

Model_testing = pd.read_csv('Battery_Testing_Data.csv')

Time = list(np.array(Model_testing['Time (s)'][18000:]))

N = len(Time)

V_model_testing = np.array(Model_testing['Voltage (V)'][18000:])

I_model_testing = np.array(Model_testing['Current (mA)'][18000:])*10**(-3)

T_model_testing = np.array(Model_testing['Temperature'][18000:])

## Josh parameters

R101 = 1.48113763e-01
R102 = 5.65859345e-03
b1 = 0
b2 = 1.26842729e+00
c1 = 3.30006677e-01
c2 = 1.19474291e+02
Offset = 0.0055 # 90% 0.007 60% 0.0046

R = 8.314
T0 = 293.15 #293.15  60%: 330
E = -11750  # 60% -101000

def R1_final(T,I):
    return R101*np.exp(-(I-b1)**2/c1)*np.exp(-E/R*(1/T-1/T0))+R102*np.exp(-(I-b2)**2/c2)*np.exp(-E/R*(1/T-1/T0))+Offset


R0 = 0.019554687500000018

C1 = 1959.7753319403273

z = [0 for i in range(N)]

z[0] = 89.62

V_model = [0 for i in range(N)]

I_model = -I_model_testing

I_R1 = [0 for i in range(N)]

I_R1[0] = 0


T_amb = 20

T_cell = [0 for i in range(N)]
T_cell[0] = 19.82

Radius = 18.33*10**(-3)/2
Length = 64.85*10**(-3)
A = 2*np.pi*Radius*(Radius+Length)
m = 45*10**(-3)
cp = 825

h = 30

R1 = R1_final(T_cell[0],I_model[0])

for i in range(N):
    SOCindex = round((SOC[0]-z[i])/SOC_step)
    OCV[SOCindex]
    R1 = R1_final(T_cell[i],I_model[i])
    V_model[i] = OCV[SOCindex]-R1*I_R1[i]-R0*I_model[i]
    if i!=N-1:
        Deltat = Time[i+1]-Time[i]
        z[i+1] = z[i]-I_model[i]*(Deltat)*100/(Q*3600)
        I_R1[i+1] = np.exp(-Deltat/(R1*C1))*I_R1[i]+(1-np.exp(-Deltat/(R1*C1)))*I_model[i]
        T_cell[i+1] = T_cell[i]+(R1*I_R1[i]**2+R0*I_model[i]**2-h*A*(T_cell[i]-T_amb))/(m*cp)

plt.figure(1)
plt.plot(Time,T_model_testing,color='red')
plt.plot(Time, T_cell)
#plt.plot(Time,R0_contr,color='blue', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,R1_contr,color='green', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,Q_contr,color='purple', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend(['Temperature - testing data','Temperature - modelisation'])
plt.show()


plt.figure(2)
plt.plot(Time,I_model)
plt.plot(Time,I_R1,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend(['Training data','I_R1'])
plt.show()

plt.figure(3)
plt.plot(Time,V_model_testing)
plt.plot(Time,V_model,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(['Voltage - testing data','Voltage - modelisation'])
plt.show()