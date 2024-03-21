import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def R0Temp():
    avgTemp = np.array([273.15,293.15,313.15])
    avgR0 = np.array([0.04026,0.01987,0.01499])

    R = 8.314
    E = -19500

    temp = np.linspace(260,330,200)
    R0 = avgR0[1]*np.exp(-E/R*(1/temp-1/avgTemp[1]))

    plt.scatter(avgTemp,avgR0)
    plt.plot(temp,R0,'--k')
    plt.xlabel('Temperature (K)')
    plt.ylabel('R$_{0}$ (\u03A9)')
    plt.show()

def R1SOC():

    SOC = np.arange(90,10,-10)
    R1 = np.array([0.01761,0.01841,0.01801,0.01500,0.01481,0.01321,0.01841,0.03122])

    poly = np.poly1d(np.polyfit(SOC,R1,2))
    x = np.linspace(0,100,100)

    plt.plot(x,poly(x),'--k')
    plt.scatter(SOC,R1)
    plt.xlabel('SOC (%)')
    plt.ylabel('R$_{1}$ (\u03A9)')
    plt.show()

def R1Temp():
    avgTemp = np.array([273.15,293.15,313.15])

    # 20A discharge
    # avgR1_60 = np.array([0.01020, 0.00600, 0.00500])
    # avgR1_30 = np.array([0.02240, 0.00650, 0.00505])
    # avgR1_90 = np.array([0.01205, 0.00800, 0.00640])

    # 2.5A discharge
    avgR1_60 = np.array([0.01500, 0.01200, 0.00761])
    avgR1_30 = np.array([0.01841,0.00800,0.00520])
    avgR1_90 = np.array([0.01761,0.00800,0.00680])

    R = 8.314
    # E = -12000

    temp = np.linspace(260,330,200)

    def R1_temp(temp,E):
        return avgR1_90[1]*np.exp(-E/R*(1/temp-1/avgTemp[1]))

    E, popc = curve_fit(R1_temp, avgTemp, avgR1_90, maxfev=99999)
    print(E)

    plt.scatter(avgTemp,avgR1_60)
    plt.plot(temp,R1_temp(temp,E),'--k')
    plt.xlabel('Temperature (K)')
    plt.ylabel('R$_{1}$ (\u03A9)')
    plt.ylim([0,0.02])
    # plt.title('60%')
    plt.show()

    R1 = avgR1_30[1] * np.exp(-E / R * (1 / temp - 1 / avgTemp[1]))

    plt.scatter(avgTemp, avgR1_30)
    plt.plot(temp, R1, '--k')
    plt.xlabel('Temperature (K)')
    plt.ylabel('R$_{1}$ (\u03A9)')
    plt.ylim([0, 0.024])
    # plt.title('30%')
    plt.show()

    R1 = avgR1_90[1] * np.exp(-E / R * (1 / temp - 1 / avgTemp[1]))

    plt.scatter(avgTemp, avgR1_90)
    plt.plot(temp, R1, '--k')
    plt.xlabel('Temperature (K)')
    plt.ylabel('R$_{1}$ (\u03A9)')
    plt.ylim([0, 0.02])
    # plt.title('90%')
    plt.show()

def R1Temp_CurrentGauss():
    R101 = 0.005
    R102 = 0.003
    b1 = -0.5
    b2 = b1
    c1 = 60
    c2 = 30
    Offset = 0.0055

    # SOC 60%
    I_model = np.array([-2.5,1.25,-5,2.5,-10,4,-20,4])
    R1_Model0C = np.array([0.0120,0.0160,0.0110,0.0176,0.0109,0.0180,0.0102,0.0202])
    R1_Model20C = np.array([0.0080,0.0080,0.0060,0.0080,0.0060,0.0075,0.0055,0.0050])
    R1_Model40C = np.array([0.0056,0.0056,0.0062,0.0068,0.0053,0.0062,0.0046,0.0035])

    I = np.linspace(-20,4,200)
    R = 8.314
    T0 = 293.15
    E = np.array([-13500,-13500,-13500])

    T = np.array([273.15, 293.15, 313.15])

    def R1_final(I, R101, b1, c1, R102, b2, c2):
        res = 0
        return R101 * np.exp(-(I - b1) ** 2 / c1) + R102 * np.exp(-(I - b2) ** 2 / c2)  # +Offset

    I = np.linspace(-20,4,200)

    popt0,pcov0 = curve_fit(R1_final,I_model,R1_Model0C)
    print(popt0)

    R1_0C = R1_final(I,*popt0)
    # R1_20C = R1_final(I, 1)
    # R1_40C = R1_final(I, 2)

    plt.scatter(I_model, R1_Model0C,color='b')
    # plt.scatter(I_model, R1_Model20C, color='g')
    # plt.scatter(I_model, R1_Model40C, color='r')
    plt.plot(I, R1_0C, '--b')
    # plt.plot(I, R1_20C, '--g')
    # plt.plot(I, R1_40C, '--r')

    plt.xlabel('Current pulse (A)')
    plt.ylabel('R$_{1}$ (\u03A9)')
    # # plt.ylim([0, 0.025])
    plt.legend(['0\xb0C','20\xb0C','40\xb0C'])
    plt.show()

def R1Temp_Current():

    # R101 = 5.89893648e-03
    # R102 = 8.94572149e+01
    # b1 = 0
    # b2 = 0.002
    # c1 = 1.50345355e-01
    # c2 = 3.21368776e-01
    # Offset = 0.0055 # 90% 0.007 60% 0.0046
    R101 = 1.48113763e-01
    R102 = 5.65859345e-03
    b1 = 0
    b2 = 1.26842729e+00
    c1 = 3.30006677e-01
    c2 = 1.19474291e+02
    Offset = 0.0055

    I = -20
    R = 8.314
    T0 = 293.15 #293.15  60%: 330
    E = np.array([-31500,-14500,-11750])  # 60% -101000

    avgTemp = np.array([273.15, 293.15, 313.15])
    avgR1 = np.array([[0.01020, 0.00600, 0.00500],
                      [0.02240, 0.00650, 0.00505],
                      [0.01205, 0.00800, 0.00640]])

    def R1_final(T,SOC_index):

        return R101*np.exp(-(I-b1)**2/c1)*np.exp(-E[SOC_index]/R*(1/T-1/T0))+R102*np.exp(-(I-b2)**2/c2)*np.exp(-E[SOC_index]/R*(1/T-1/T0))+Offset

    T = np.linspace(100,330,200)

    for SOC_index in range(3):
        R1 = R1_final(T,SOC_index)

        plt.scatter(avgTemp, avgR1[SOC_index])
        plt.plot(T, R1, '--k')
        plt.xlabel('Temperature (K)')
        plt.ylabel('R$_{1}$ (\u03A9)')
        plt.ylim([0, 0.025]) #([0, 0.025])
        plt.xlim([100,340])
        plt.show()


# R0Temp()
# R1SOC()
R1Temp()
# R1Temp_CurrentGauss()
# R1Temp_Current()