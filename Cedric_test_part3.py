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

T_model_testing = np.array(Model_testing['Temperature'][18000:])+273.15

## Josh parameters

R0_average = 0.019554687500000018

C1_average = 1962.9469051907095

# R101 = -119.02362263 #0.07           #@0: 0.06, @20: 0.07, @40: 0.05
# R102 = 119.14161295 #0.006        #@0: 0.017, @20: 0.006, @40: 0.003
# b1 = 0 #-0.5         #@0: -0.5, @20: -0.5, @40: -0.5
# b2 = b1
# c1 = 0.55169725#1          #@0: 0.5, @20: 1, @40: 1
# c2 = 0.55169743#30           #@0: 70, @20: 30, @40: 60
# Offset = 0.0055      #@0: 0.025, @20: 0.0055, @40: 0

R101 = 1.48113763e-01
R102 = 5.65859345e-03
b1 = 0
b2 = 1.26842729e+00
c1 = 3.30006677e-01
c2 = 1.19474291e+02
Offset = 0.0055 # 90% 0.007 60% 0.0046

R = 8.314
T0 = 293.15 #293.15  60%: 330
E = -12825  # 60% -101000

def R1_final(T,I):
    return R101*np.exp(-(I-b1)**2/c1)*np.exp(-E/R*(1/T-1/T0))+R102*np.exp(-(I-b2)**2/c2)*np.exp(-E/R*(1/T-1/T0))+Offset

R01 = R0_average

def R0_final(T):
    return R01*np.exp(-E/R*(1/T-1/T0))


C1 = C1_average

z = [0 for i in range(N)]

z[0] = 89.62

V_model = [0 for i in range(N)]

I_model = -I_model_testing

I_R1 = [0 for i in range(N)]

I_R1[0] = 0

T_amb = 293.15

T_cell = [0 for i in range(N)]
T_cell[0] = 19.82+273.15

Radius = 18.33*10**(-3)/2
Length = 64.85*10**(-3)
A = 2*np.pi*Radius*(Radius+Length)
m = 45*10**(-3)
cp = 825

h = 35

R1 = R1_final(T_cell[0],I_model[0])
R0 = R0_final(T_cell[0])

for i in range(N):
    SOCindex = round((SOC[0]-z[i])/SOC_step)
    OCV[SOCindex]
    if I_model[i]!=0:
        R1 = R1_final(T_cell[i],I_model[i])
    R0 = R0_final(T_cell[i])
    V_model[i] = OCV[SOCindex]-R1*I_R1[i]-R0*I_model[i]
    if i!=N-1:
        Deltat = Time[i+1]-Time[i]
        z[i+1] = z[i]-I_model[i]*(Deltat)*100/(Q*3600)
        I_R1[i+1] = np.exp(-Deltat/(R1*C1))*I_R1[i]+(1-np.exp(-Deltat/(R1*C1)))*I_model[i]
        T_cell[i+1] = T_cell[i]+(R1*I_R1[i]**2+R0*I_model[i]**2-h*A*(T_cell[i]-T_amb)*Deltat)/(m*cp)

plt.figure(1)
plt.plot(Time,T_model_testing,color='red')
plt.plot(Time, T_cell)
#plt.plot(Time,R0_contr,color='blue', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,R1_contr,color='green', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,Q_contr,color='purple', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend(['Testing data','Model'])
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
error = (V_model-V_model_testing)/V_model_testing*100
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
    Error_total += abs(V_model_testing[k]-V_model[k])

Error_total = Error_total/N
print('Error_total=',Error_total)

plt.figure(4)
plt.plot(Time,I_model)
plt.plot(Time,I_R1,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend(['testing data','I_R1'])
plt.show()

plt.figure(5)
plt.plot(Time,V_model_testing)
plt.plot(Time,V_model,color='red', linestyle='dashed', linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend(['Testing data','Model'])
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
error = (T_cell-T_model_testing)/T_model_testing*100
ax1.plot(Time,error,'b-')
ax2.plot(Time,I_model,'g-')
#plt.plot(Time,R1_cont,color='green', linestyle='dashed', linewidth=0.5)
#plt.plot(Time,I_R1,color='red', linestyle='dashed', linewidth=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Percentage Error - Temperature (%)', color='b')
ax1.set_ylim([-0.6,0.6])
ax2.set_ylabel('Current (A)', color='g')
ax2.set_ylim([-15.5,15.5])
plt.show()

plt.figure(7)
plt.plot(Time,error,'b-')
plt.xlabel('Time (s)')
plt.ylabel('Percentage Error - Temperature (%)', color='b')
plt.show()