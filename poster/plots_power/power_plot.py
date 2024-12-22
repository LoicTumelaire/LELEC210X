import numpy as np
import matplotlib.pyplot as plt

_, time, v_res, v_sound = np.loadtxt('micro_results.csv', delimiter=',', unpack=True, skiprows=8)
# remove where time inferior to 1sec and superior to 3.6sec
v_res = v_res[(time >= 1) & (time <= 3.6)]
v_sound = v_sound[(time >= 1) & (time <= 3.6)]
time = time[(time >= 1) & (time <= 3.6)]

time = time - time[0] # start time at 0


R = 1.8e3
power = 3.3*v_res/R*1e6 # in uW

#plot power and sound on different plot with shared x-axis
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plot power
axs[0].plot(time, power, label='Power', color='blue', alpha=0.8)
axs[0].set_ylabel('Power ($\mu$W)')
axs[0].legend()
axs[0].grid()

# Plot v_sound
axs[1].plot(time, v_sound, label='Sound Voltage', color='green', alpha=0.8)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Sound Voltage (V)')
axs[1].legend()
axs[1].grid()

# Adjust layout
plt.tight_layout()
plt.savefig('micro_power.pdf', bbox_inches='tight')
plt.clf()
plt.close()

#static consumption with voltage

voltage = np.array([2, 2.4, 2.8, 3.2, 3.3, 3.4])
voltage_resistor = np.array([70.9, 70.91, 73, 74.59, 76, 76.17])*1e-3 # in V
current = voltage_resistor/R
power = current*voltage*1e6 # in uW

plt.plot(voltage, power, label='Power', color='blue', alpha=0.8)
plt.xlabel('Voltage (V)')
plt.ylabel('Power ($\mu$W)')
plt.legend()
plt.grid()
plt.savefig('micro_power_voltage.pdf', bbox_inches='tight')
plt.clf()
plt.close()

# saturation with lower voltage
_, time, v_res, v_sound = np.loadtxt('micro_results_2V.csv', delimiter=',', unpack=True, skiprows=8)
# remove where time inferior to 1sec and superior to 3.6sec
v_res = v_res[(time >= 0.5) & (time <= 2.5)]
v_sound = v_sound[(time >= 0.5) & (time <= 2.5)]
time = time[(time >= 0.5) & (time <= 2.5)]
time = time - time[0] # start time at 0

plt.plot(time, v_sound, label='Sound Voltage', color='green', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Sound Voltage (V)')
plt.legend()
plt.grid()
plt.savefig('micro_power_saturation.pdf', bbox_inches='tight')
plt.clf()
plt.close()

#Radio power plot
_, time_radio, v_radio = np.loadtxt('scopy radio.csv', delimiter=',', unpack=True, skiprows=8)
v_radio = v_radio[(time_radio >= 3.549+time_radio[0]) & (time_radio <= 4.78+time_radio[0])]
time_radio = time_radio[(time_radio >= 3.549+time_radio[0]) & (time_radio <= 4.78+time_radio[0])]
time_radio = time_radio - time_radio[0] # start time at 0

_, time_radio_15, v_radio_15 = np.loadtxt('scopy_radio_-15db.csv', delimiter=',', unpack=True, skiprows=8)
v_radio_15 = v_radio_15[(time_radio_15 >= 8.897+time_radio_15[0]) & (time_radio_15 <= 10.128+time_radio_15[0])]
time_radio_15 = time_radio_15[(time_radio_15 >= 8.897+time_radio_15[0]) & (time_radio_15 <= 10.128+time_radio_15[0])]
time_radio_15 = time_radio_15 - time_radio_15[0] # start time at 0

_, time_radio_30, v_radio_30 = np.loadtxt('scopy_radio_-30db.csv', delimiter=',', unpack=True, skiprows=8)
v_radio_30 = v_radio_30[(time_radio_30 >= 9.6+time_radio_30[0]) & (time_radio_30 <= 10.831+time_radio_30[0])]
time_radio_30 = time_radio_30[(time_radio_30 >= 9.6+time_radio_30[0]) & (time_radio_30 <= 10.831+time_radio_30[0])]
time_radio_30 = time_radio_30 - time_radio_30[0] # start time at 0

R = 27
power_radio = 3.3*v_radio/R*1e3 # in mW
power_radio_15 = 3.3*v_radio_15/R*1e3 # in mW
power_radio_30 = 3.3*v_radio_30/R*1e3 # in mW

plt.plot(time_radio, power_radio, label='Power at Tx=0dBm', color='blue', alpha=0.8)
plt.plot(time_radio_15, 3.3*v_radio_15/R*1e3, label='Power at Tx=-15dBm', color='red', alpha=0.8)
plt.plot(time_radio_30, 3.3*v_radio_30/R*1e3, label='Power at Tx=-30dBm', color='green', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.grid()
plt.savefig('radio_power.pdf', bbox_inches='tight')
plt.clf()
plt.close()

print('Duration : ', time_radio[-1])
print('Average radio power : ', np.mean(power_radio), ' mW')

#MCU power plot
_, time_mcu_lower_clock, v_mcu_lower_clock = np.loadtxt('scopy_mesure_clock_24Mhz.csv', delimiter=',', unpack=True, skiprows=8)
v_mcu_lower_clock = v_mcu_lower_clock[(time_mcu_lower_clock >= 2.826+time_mcu_lower_clock[0]) & (time_mcu_lower_clock <= 5.422+time_mcu_lower_clock[0])]
time_mcu_lower_clock = time_mcu_lower_clock[(time_mcu_lower_clock >= 2.826+time_mcu_lower_clock[0]) & (time_mcu_lower_clock <= 5.422+time_mcu_lower_clock[0])]
time_mcu_lower_clock = time_mcu_lower_clock - time_mcu_lower_clock[0] # start time at 0

_, time_mcu, v_mcu = np.loadtxt('scopy mesures_mcu.csv', delimiter=',', unpack=True, skiprows=8)
v_mcu = v_mcu[(time_mcu >= 0.876+time_mcu[0]) & (time_mcu <= 3.285+time_mcu[0])]
time_mcu = time_mcu[(time_mcu >= 0.876+time_mcu[0]) & (time_mcu <= 3.285+time_mcu[0])]
time_mcu = time_mcu - time_mcu[0] # start time at 0

R = 27
power_mcu_lower_clock = 3.3*v_mcu_lower_clock/R*1e3 # in mW
power_mcu = 3.3*v_mcu/R*1e3 # in mW

plt.plot(time_mcu_lower_clock, power_mcu_lower_clock, label='Sys_CLK=24Mhz', color='blue', alpha=0.8)
plt.plot(time_mcu, power_mcu, label='Sys_CLK=48Mhz', color='red', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.grid()
plt.show()

#compute mean between 0.8 sec and 2.2 sec
mean_mcu = np.mean(power_mcu[(time_mcu >= 0.7942) & (time_mcu <= 2.0272)])
mean_mcu_lower_clock = np.mean(power_mcu_lower_clock[(time_mcu_lower_clock >= 0.7942) & (time_mcu_lower_clock <= 2.1985)])
print('Average power of mcu with 48Mhz : ', mean_mcu, ' mW')
print('Average power of mcu with 24Mhz : ', mean_mcu_lower_clock, ' mW')

#recover again to do different plot
_, time_mcu, v_mcu = np.loadtxt('scopy mesures_mcu.csv', delimiter=',', unpack=True, skiprows=8)
v_mcu = v_mcu[(time_mcu >= 1.675+time_mcu[0]) & (time_mcu <= 2.91+time_mcu[0])]
time_mcu = time_mcu[(time_mcu >= 1.675+time_mcu[0]) & (time_mcu <= 2.91+time_mcu[0])]
time_mcu = time_mcu - time_mcu[0] # start time at 0

R = 27
power_mcu = 3.3*v_mcu/R*1e3 # in mW

print('Duration : ', time_mcu[-1])
print('Average power of mcu: ', np.mean(power_mcu), ' mW')

# plot radio and MCU power on the same plot
power_afe_micro = 140*1e-3*np.ones_like(time_mcu) # in mW
sum = power_mcu[:len(time_radio)] + power_afe_micro[:len(time_radio)] + power_radio

plt.plot(time_radio, power_radio, label='Radio', color='blue', alpha=0.8)
plt.plot(time_mcu, power_afe_micro, label='AFE', color='green', alpha=0.8)
plt.plot(time_mcu, power_mcu, label='MCU', color='red', alpha=0.8)
plt.plot(time_radio, sum, label='Whole system', color='black', alpha=0.65, linewidth=0.45)
plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.grid()
plt.savefig('global_power_profile.pdf', bbox_inches='tight')
plt.clf()
plt.close()

#mean power :
print('Total mean power : ', np.mean(sum), ' mW')

#integrate each power to get the energy
energy_radio = np.trapezoid(power_radio, time_radio)
energy_mcu = np.trapezoid(power_mcu, time_mcu)
energy_afe_micro = power_afe_micro[0]*(time_mcu[-1] - time_mcu[0])

data = [energy_radio, energy_mcu, energy_afe_micro]
labels = ['Radio', 'MCU', 'AFE']
custom_colors = ['blue', 'red', 'green']
# Custom function to display actual values
def actual_values(pct, all_values):
    absolute = round(pct/100.*np.sum(all_values), 2)
    return f'{absolute} mJ'

# Plot the pie chart
plt.pie(
    data, 
    labels=labels, 
    colors=custom_colors,
    autopct=lambda pct: actual_values(pct, data),
    wedgeprops={"alpha": 0.8}
)
plt.savefig('energy_pie.pdf', bbox_inches='tight')
plt.clf()
plt.close()
