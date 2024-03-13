import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares

## part 1

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
    plt.ylabel('Absolute Error (V)')
    plt.show()

    # overlaying current
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Absolute Error (V)', c='b')
    ax1.plot(t,err,'b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim([-0.2,0.2])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Current (mA)', c='g')
    ax2.plot(t,I,'g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim([-16000, 16000])

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

## part 2

def FirstOrderECN(t,R0,R1,C1,I,dt,OCV):
    # Initialize i_R1
    i_R1 = np.zeros(len(t))

    # Simulate i_R1 evolution
    for j in range(1, len(t)):
        i_R1[j] = np.exp(-dt/(R1*C1))*i_R1[j-1] + (1-np.exp(-dt/(R1*C1)))*I[j-1]

    #Calculate cell voltage
    modelvol = OCV-(R1*i_R1)-(R0*I)
    return modelvol

def FitFirstOrderECN(t, I, V, dt, OCV):
    # Objective function to minimize
    def objective_function(t, params, I, V, dt, OCV):
        R0, R1, C1 = params
        predvoltage = FirstOrderECN(t, R0, R1, C1, I, dt, OCV)
        residual = predvoltage - V
        return residual

    # Parameters initial guesses (R0, R1, C1)
    InitialGuess = (1, 1, 1)

    # Define the objective function with fixed dt and OCV
    obj_func = lambda params: objective_function(t, params, I, V, dt, OCV)
    result = least_squares(obj_func, InitialGuess)

    # Extract the optimized parameters
    optimized_params = result.x
    return optimized_params

# Task 1 Model Updated
def ECNModel(temp, npulse, nSOC, plots=0):

    if temp == 0:
        last_pulse = 6
    else:
        last_pulse = 8

    # Read CSV file containing training data @ specified temperature
    file = 'Model_Training_Data_'+str(temp)+'.csv'
    df = pd.read_csv(file)
    t = np.array(df['Time (s)'][:])
    I = np.array(df['Current (A)'][:])
    V = np.array(df['Voltage (V)'][:])

    if plots:
        # Plot Data - Current
        plt.figure(1, figsize=(10, 6))
        plt.plot(t, I, label='Current (A)', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Discharge Pulses')
        plt.grid(True)

        # Plot Data - Voltage
        plt.figure(2, figsize=(10, 6))
        plt.plot(t, V, label='Voltage (V)', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Discharge Pulses')
        plt.grid(True)

    def identifypulses(I, threshold=0.5, pulse_duration=10):
        pulses = []
        in_pulse = False
        pulse_start = 0

        for i in range(1, len(I)):
            if abs(I[i] - I[i - 1]) > threshold:
                if not in_pulse:
                    pulse_start = i - 1
                    in_pulse = True
            elif in_pulse:
                if i - pulse_start >= pulse_duration * 100:  # Check if pulse duration is approximately 10 seconds
                    pulses.append((pulse_start, i))
                    in_pulse = False
        return pulses

    # Analyze pulses for the full data range
    pulses = identifypulses(I)
    pulsestimes = []

    for start, end in pulses:
        tpulsestart = t[start]
        tpulsesend = t[end - 1]
        pulsestimes.append((tpulsestart, tpulsesend))

    # Divide list: 8 subsections - 8 SOC levels
    nSOCs = 8
    nPulsesSOC = 8
    SOCpulseintervals = [[] for _ in range(nSOCs)]

    for i, (tstart, tend) in enumerate(pulsestimes):
        SOCindex = i // nPulsesSOC
        SOCpulseintervals[SOCindex].append((tstart, tend))

    # Print the identified pulse intervals for each SOC range
    if plots:
        for idx, pulses in enumerate(SOCpulseintervals):
            SOCstart = 90 - idx * 10
            SOCend = 100 - idx * 10
            print(f"SOC Range: {SOCstart}% - {SOCend}%")
            for pulse_idx, (start, end) in enumerate(pulses):
                print(f"Pulse {pulse_idx + 1}: Start Time: {start}, End Time: {end}")

    # SOCind: list to hold results
    SOCind = []

    pp1 = 4
    pp2 = 13

    for i in range(8):
        if i == 7:
            lp = last_pulse
        else:
            lp = 8
        SOC = SOCpulseintervals[i]
        Pulse = []
        plt.figure(pp1)
        plt.figure(pp2)
        tpulse = np.zeros((1000, lp))
        Ipulse = np.zeros((1000, lp))
        Vpulse = np.zeros((1000, lp))
        for j in range(lp):
            pulserange = SOC[j]
            # Use NumPy indexing to find matching time indices
            k_indices = np.where((t >= pulserange[0]) & (t <= pulserange[1]))[0]
            # Populate corresponding pulse data
            tpulse[:len(k_indices), j] = t[k_indices]
            Ipulse[:len(k_indices), j] = I[k_indices]
            Vpulse[:len(k_indices), j] = V[k_indices]

            if plots:
                # Plot
                plt.figure(pp1)
                plt.plot(tpulse[:, j], Ipulse[:, j], label=f'Pulse {j + 1}')
                plt.title(f'SOC {i}')
                plt.xlabel('Time (s)')
                plt.ylabel('Current (A)')
                plt.legend()
                plt.figure(pp2)
                plt.plot(tpulse[:, j], Vpulse[:, j], label=f'Pulse {j + 1}')
                plt.title(f'SOC {i}')
                plt.xlabel('Time (s)')
                plt.ylabel('Voltage (V)')
                plt.legend()

            Pulse.append([j, np.copy(tpulse[:, j]), np.copy(Ipulse[:, j]), np.copy(Vpulse[:, j])])
        pp1 = pp1 + 1
        pp2 = pp2 + 1
        SOCind.append(Pulse)

    if plots:
        plt.figure(3)
        plt.plot(tpulse[:, 0], Vpulse[:, 0], label=f'Pulse {j + 1}', color='red')
        plt.title(f'SOC {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()

        fig, ax = plt.subplots()
        plt.plot(tpulse[:, 0], Ipulse[:, 0], label=f'Pulse {j + 1}', color='red')
        plt.title(f'SOC {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.legend()

    SOCinddata = SOCind[(nSOC)]
    SOCinddata = SOCinddata[(npulse)]
    t = SOCinddata[1]
    I = SOCinddata[2]
    V = SOCinddata[3]

    # read csv containing OCV data
    df = pd.read_csv('SOC_OCV_MFCTT_2019.csv')

    SOC_raw = np.array(df['SOC'][:])
    OCV_raw = np.array(df['Ecell/V'][:])

    dSOC = SOC_raw[0]-SOC_raw[1] # SOC step
    dt = t[1]-t[0] # time step

    # Immediate voltage jump in current pulse application
    # Find the maximum/minimum voltage during the pulse
    V_max = max(V)
    V_min = min(V)

    # Find voltage at the moment when current switched to zero
    for i in range(1, len(I)):
        if I[i] == 0 and I[i-1] != 0:
          V_switch = V[i]
          t_switch = t[i]
          break
        else:
          # If loop doesn't find zero current, use first voltage
          V_switch = V[0]
          t_switch = t[0]

    dv0 = abs(V_max-V_switch) if abs(V_max-V_switch) > abs(V_min-V_switch) else abs(V_min-V_switch)

    # Calculate current difference
    di = abs(max(I)-min(I))
    R0 = dv0 / di

    # Find index corresponding to t_switch
    idx_switch = next(i for i, t_val in enumerate(t) if t_val >= t_switch)

    # Find the index of the first voltage value after the current is switched off
    idx_steady_state = None
    steady_state_duration_threshold = 25
    for i in range(idx_switch, len(V)-1):
        if all(abs(V[j] - V[j+1]) < 0.001 for j in range(i, min(i + int(steady_state_duration_threshold / dt), len(V)-1))):
        #if abs(V[i] - V[i+1]) < 0.001:
            idx_steady_state = i
            break
    if idx_steady_state is not None:
        V_steady_state = V[idx_steady_state]
    else:
        # Handle the case where steady state is not found
        V_steady_state = None

    dvinf = abs(V_steady_state-V_max) if abs(V_steady_state-V_max) > abs(V_steady_state-V_min) else abs(V_steady_state-V_min)
    R1 = (dvinf / di) - R0


    # Time to settle after pulse application
    Vmax = V[V==V_max]
    tmax = t[V==Vmax[0]]
    Vmin = V[V==V_min]
    tmin = t[V==Vmin[0]]
    dt_steady_state = t[idx_steady_state]-tmax[0] if abs(V_steady_state-V_max) > abs(V_steady_state-V_min) else t[idx_steady_state]-tmin[0]
    C1 = dt_steady_state / (4 * R1)

    Parameters = [R0,R1,C1]

    if plots:
        plt.figure()
        plt.plot(t, V, color='blue')
        plt.scatter(t_switch, V_switch, color='yellow')
        if len(V[V==V_max])<len(V[V==V_min]):
          plt.scatter(tmax[0], V_max, color='red')
        else:
          plt.scatter(tmin[0], V_min, color='red')
        plt.scatter(t[idx_steady_state], V_steady_state, color='green')
        plt.title(f'SOC {nSOC} Pulse {npulse}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()

        print("Parameters (First Order ECN):", R0, R1, C1)

    Qn = 2500 # nominal capacity in mAh

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

    # Calculate model voltage
    V_model = FirstOrderECN(t, R0, R1, C1, I, dt, OCV) if abs(V_steady_state-V_max) > abs(V_steady_state-V_min) else FirstOrderECN(t, -R0, -R1, -C1, I, dt, OCV)

    if plots:
        fig, ax = plt.subplots()
        plt.plot(t,V,'k')
        plt.plot(t,V_model,'r')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend(['Experimental','Model'])
        plt.show()

        err = V-V_model # Error between experimental and model voltage
        plt.plot(t,err)
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error')
        plt.show()

        # Overlaying current
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

    return np.array(Parameters)

temp = 0

if temp == 0:
    last_pulse = 6
else:
    last_pulse = 8
AllSOCParam = np.ndarray([8*7+last_pulse,3])
AverageParam = []
for i in range(8):
    if i == 7:
        lp = last_pulse
    else:
        lp = 8
    for j in range(lp):
        AllSOCParam[i*8+j] = ECNModel(temp,j,i)
        # if i == 3 & j == 1:
        #     print(ECNModel(temp,j,i))
# print(AllSOCParam)

np.savetxt('data.csv', (AllSOCParam[:,0],AllSOCParam[:,1],AllSOCParam[:,2]), delimiter=',')

# ThModel()
# Temp_Model()