import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def R0Temp():
    avgTemp = np.array([293.15,313.15,333.15])
    avgR0 = np.array([0.0402621638540363,0.021,0.0189375])

    R = 8.314
    E = -18000

    temp = np.linspace(280,350,200)
    R0 = avgR0[1]*np.exp(-E/R*(1/temp-1/avgTemp[1]))

    plt.scatter(avgTemp,avgR0)
    plt.plot(temp,R0,'--k')
    plt.xlabel('Temperature (\xb0C)')
    plt.ylabel('Resistance$_{0}$ (\u03A9)')
    plt.show()

def R1SOC():

    SOC = np.arange(90,10,-10)
    R1 = np.array([0.01600,0.01480,0.01440,0.01200,0.01120,0.01120,0.01640,0.02920])

    poly = np.poly1d(np.polyfit(SOC,R1,3))
    x = np.linspace(20,90,100)

    plt.plot(x,poly(x),'--k')
    plt.scatter(SOC,R1)
    plt.xlabel('SOC (%)')
    plt.ylabel('Resistance$_{1}$ (\u03A9)')
    plt.show()

def R1Temp():
    avgTemp = np.array([293.15,313.15,333.15])
    avgR1_60 = np.array([0.01200,0.00800,0.00560])
    avgR1_30 = np.array([0.01640, 0.00800, 0.00360])
    avgR1_90 = np.array([0.01600, 0.00800, 0.00600])

    R = 8.314
    E = -15500

    temp = np.linspace(280,350,200)
    R1 = avgR1_60[1]*np.exp(-E/R*(1/temp-1/avgTemp[1]))

    plt.scatter(avgTemp,avgR1_60)
    plt.plot(temp,R1,'--k')
    plt.xlabel('Temperature (\xb0C)')
    plt.ylabel('Resistance$_{1}$ (\u03A9)')
    plt.ylim([0,0.015])
    # plt.title('60%')
    plt.show()

    plt.scatter(avgTemp, avgR1_30)
    plt.plot(temp, R1, '--k')
    plt.xlabel('Temperature (\xb0C)')
    plt.ylabel('Resistance$_{1}$ (\u03A9)')
    plt.ylim([0, 0.02])
    # plt.title('30%')
    plt.show()

    plt.scatter(avgTemp, avgR1_90)
    plt.plot(temp, R1, '--k')
    plt.xlabel('Temperature (\xb0C)')
    plt.ylabel('Resistance$_{1}$ (\u03A9)')
    plt.ylim([0, 0.02])
    # plt.title('90%')
    plt.show()





# R0Temp()
# R1SOC()
R1Temp()