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
    df0 = pd.read_csv('Model_Training_Data_0.csv')

    t0 = np.array(df0['Time (s)'][:])
    I0 = np.array(df0['Current (A)'][:])
    V0 = np.array(df0['Voltage (V)'][:])

    # read csv containing model data
    df20 = pd.read_csv('Model_Training_Data_20.csv')

    t20 = np.array(df20['Time (s)'][:])
    I20 = np.array(df20['Current (A)'][:])
    V20 = np.array(df20['Voltage (V)'][:])

    # read csv containing model data
    df40 = pd.read_csv('Model_Training_Data_40.csv')

    t40 = np.array(df40['Time (s)'][:])
    I40 = np.array(df40['Current (A)'][:])
    V40 = np.array(df40['Voltage (V)'][:])

    # interested in discharge data
    # t = t[18000:]
    # I = -I[18000:]
    # V = V[18000:]

    plt.plot(t0,I0)
    plt.plot(t20, I20)
    plt.plot(t40, I40)
    plt.ylabel('Current')
    plt.legend(['0C', '20C', '40C'])
    plt.show()
    plt.plot(t0,V0)
    plt.plot(t20, V20)
    plt.plot(t40, V40)
    plt.ylabel('Voltage')
    plt.legend(['0C','20C','40C'])
    plt.show()

# ThModel()
Temp_Model()