import numpy as np
import matplotlib.pyplot as plt

_, time, volt = np.loadtxt('o3_compilation_optimisation.csv', delimiter=',', unpack=True, skiprows=8)
_, time2, volt2 = np.loadtxt('scopy mesures_mcu.csv', delimiter=',', unpack=True, skiprows=8)

#keep only time between 4.95 and 6.195
volt = volt[(time >= 4.95) & (time <= 6.195)]
time = time[(time >= 4.95) & (time <= 6.195)]

#keep between 6.616 and 7.867
volt2 = volt2[(time2 >= 6.616) & (time2 <= 7.867)]
time2 = time2[(time2 >= 6.616) & (time2 <= 7.867)]

# start time at 0
time = time - time[0]
time2 = time2 - time2[0]

R = 27
Vdd = 3.3

power = volt/R*Vdd*1000
power2 = volt2/R*Vdd*1000

plt.plot(time, power, color='blue', alpha=0.8, label="Normal compilation")
plt.plot(time2, power2, color='red', alpha=0.8, label="O3 compilation")
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.show()
#plt.savefig('o3_power.pdf', bbox_inches='tight')

# lower sample frequency

_, time, volt = np.loadtxt('f_sample_2khz.csv', delimiter=',', unpack=True, skiprows=8)
power = volt/R*Vdd*1000

plt.plot(time, power, color='blue', alpha=0.8, label="2kHz")
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.show()

#clk=8Mhz
_, time, volt = np.loadtxt('f=2khz_clk=8Mhz.csv', delimiter=',', unpack=True, skiprows=8)
power = volt/R*Vdd*1000

plt.plot(time, power, color='blue', alpha=0.8, label="CLK=8MHz")
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.show()

#Clk=2MHz
_, time, volt = np.loadtxt('f=2khz_clk=2Mhz.csv', delimiter=',', unpack=True, skiprows=8)
power = volt/R*Vdd*1000

plt.plot(time, power, color='blue', alpha=0.8, label="CLK=2MHz")
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.show()
