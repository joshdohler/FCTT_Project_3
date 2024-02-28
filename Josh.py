import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## task 1

def ThModel():
    # read csv containing test data
    df = pd.read_csv('Battery_Testing_Data.csv')

    t = np.array(df['Time (s)'][:])
    I = np.array(df['Current (mA)'][:])
    V = np.array(df['Voltage (V)'][:])
    Te = np.array(df['Temperature'][:])

    # interested in discharge data
    t = t[18000:]
    I = -I[18000:]
    V = V[18000:]
    Te = Te[18000:]

    # read csv containing OCV data
    df = pd.read_csv('SOC_OCV_MFCTT_2019.csv')

    SOC_raw = np.array(df['SOC'][:])
    OCV_raw = np.array(df['Ecell/V'][:])

    dSOC = SOC_raw[0]-SOC_raw[1] # SOC step
    dt = t[1]-t[0] # time step

    Qn = 2500 # nominal capacity in mAh
    R0 = 0.0187 # Thevenin resistance in Ohms

    SOC = np.zeros(np.shape(t)) # SOC at each time step in discharge

    # calculating initial SOC - will match OCV due to long hold
    found = 0
    i = 0
    while not found:
        if V[0] > OCV_raw[i]:
            if abs(V[0] - OCV_raw[i]) < abs(V[0] - OCV_raw[i-1]):
                SOC[0] = SOC_raw[i]
            else:
                SOC[0] = SOC_raw[i-1]
            found = 1
        else:
            i += 1

    for j in range(1,len(t)):
        SOC[j] = SOC[j-1] - 100*(I[j]*dt/3600)/Qn # SOC as percentage


    SOC_ind = ((SOC_raw[0]-SOC)/dSOC-3).astype(int) # array with index corresponding to current SOC value
    OCV = OCV_raw[SOC_ind] # OCV at each time step in discharge
    V_model = OCV - I/1000*R0

    plt.plot(t,V,'k')
    plt.plot(t,V_model,'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(['Experimental','Model'])
    plt.show()

    err = V-V_model # error between experimental and model voltage
    plt.plot(t,err)
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Error')
    plt.show()

    # overlaying current
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Absolute Error', c='b')
    ax1.plot(t,err,'b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Current (mA)', c='g')
    ax2.plot(t,I,'g')
    ax2.tick_params(axis='y', labelcolor='g')

    fig.tight_layout()
    plt.show()

def Temp_Model():
    # read csv containing model data
    df = pd.read_csv('Model_Training_Data_20.csv')

    t = np.array(df['Time (s)'][:])
    I = np.array(df['Current (A)'][:])
    V = np.array(df['Voltage (V)'][:])

    # interested in discharge data
    # t = t[18000:]
    # I = -I[18000:]
    # V = V[18000:]

    plt.plot(t,I)
    plt.ylabel('Current')
    plt.show()
    plt.plot(t,V)
    plt.ylabel('Voltage')
    plt.show()

# ThModel()
Temp_Model()