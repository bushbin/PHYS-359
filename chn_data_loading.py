"""
Example script showing use of mcphysics.data to load *.CHN files 
and subsequent manipulation of spinmob databoxes containing the data.

You need the companion .chn file, "Example_data.chn" to run this script.

Author: Brandon Ruffolo
"""
import mcphysics.data as m
import matplotlib.pyplot as plt

data = m.load_chn("Long_Am_Run.Chn")


# Print the hkeys (metadata fields)
print("hkeys: ")
print(data.hkeys)
print()


# Print the start time of the run
print("Start time: ")
print(data.h('start_time'))
print()

# Print the run description (if you enetered one!)
print("Run description:")
print(data.h('description')) 
print()

# Print the ckeys (data fields)
print("ckeys: ")
print(data.ckeys)
print()

# Print the channel data (array spanning from 0 to 2047, since the MCA is 11-bit)
print("Channel data: ")
print(data['Channel'])
print()


plt.figure()
plt.plot(data['Channel'],data['Counts'])
plt.xlabel("Channel Number", fontsize=14); plt.ylabel("Counts", fontsize=14)
plt.show()
plt.show(block=True)

