# Thermal-sims
#Single segment analysis

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:35:07 2026

@author: jamessquire
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


duration = 10 #duration s 
dt = 0.0001 #timestep s 


t = 0.7e-3     #chamber wall thickness m 
h = 0.5e-3        #rib hight m 
w = 0.5e-3       #channel widrth m 
w_r = 0.15e-3     #rib width m 
o_t = 2e-3     #outer wall thickness m 
r = 0.5e-3     #fillet radius m
dz = 1e-3       #axial length m 

T_c = 320      #coolant temp K
T_h = 3000     #hot gass temp K
T_initial = 300 #initial temps k 
k = 150        #conductivity 

c = 900        #spesific heat capasity J/(kgK)     
rho = 2500     #density of AlSi10Mg (kg/m^3)
cte = 22e-6    #coeficent of thermal expantion K^-1
h_h = 3000        #core flow convective heat transfer coeficent 
h_c = 13000        #regen convective heat transfer coeficent 

temps = np.full(20, T_initial)     #temprature array 
temps = np.append(temps, T_c)      #colant temp in index 20
temps = np.append(temps, T_h)      #core gas temp in index 21


#convective heat flux coeficents (in temp change, to be put straight into A)
def dt_conv(x, y, z, h):   
    A = x*y           
    m = x*y*z*rho                     
    return A*h*dt/(m*c)

conv = np.zeros((5))

conv[0] = dt_conv(w,dz,t/4,h_h)       #bottom regen chamber wall segment
conv[1] = dt_conv(w_r,dz,t/4,h_h)     #bottom rib chamber wall segment
conv[2] = dt_conv(w,dz,t/4,h_c)       #top regen chamber wall segment
conv[3] = 2*dt_conv(dz,h/4,w_r,h_c)   #rib segments  (here the *2 is to represent the heat flow out of both sides of the rib)
conv[4] = dt_conv(w,dz,o_t/4,h_c)     #bottom regen outer wall segment 



#fourier number (x and y are the sides of the face being analysed)
def F_0(x, y, z, dx):                  
    A = x*y
    m = x*y*z*rho
    return k*dt*A/(m*c*dx) 
    
F = np.zeros((3,9))   #array of fouier numbers

#heat transfer constants for regen chamber wall segments 
F[0,0] = F_0(w,dz,t/4,t/4)             #radial heat transfer constant 
F[1,0] = 2*F_0(dz,t/4,w,(w+w_r)/2)     #tangential heat transfer constant (*2 is to represent the heat flow out of both sides)
F[2,0] = F_0(w,t/4,dz,dz)              #axial heat transfer constant 

#heat transfer constants for rib chamber wall segments 
F[0,1] = F_0(w_r,dz,t/4,t/4)     
F[1,1] = 2*F_0(dz,t/4,w_r,(w+w_r)/2)     
F[2,1] = F_0(w_r,t/4,dz,dz)     
    
#heat transfer constants for rib chamber wall segments (interfaced with rib segments)
F[0,2] = F_0(w_r,dz,t/4,(t/4+h/4)/2)     
F[1,2] = 2*F_0(dz,t/4,w_r,(w+w_r)/2)     
F[2,2] = F_0(w_r,t/4,dz,dz)     

#heat transfer constants for rib segments (interfaced with rib chamber wall segments)
F[0,3] = F_0(w_r,dz,h/4,(t/4+h/4)/2)         
F[2,3] = F_0(w_r,h/4,dz,dz)     

#heat transfer constants for rib segments 
F[0,4] = F_0(w_r,dz,h/4,h/4)         
F[2,4] = F_0(w_r,h/4,dz,dz)     

#heat transfer constants for rib segments (interfaced with rib outer wall segments)
F[0,5] = F_0(w_r,dz,h/4,(h/4+o_t/4)/2)      
F[2,5] = F_0(w_r,h/4,dz,dz)    

#heat transfer constants for rib outer wall segments (interfaced with rib segments)
F[0,6] = F_0(w_r,dz,o_t/4,(h/4+o_t/4)/2)     
F[1,6] = 2*F_0(dz,o_t/4,w_r,(w+w_r)/2)     
F[2,6] = F_0(w_r,o_t/4,dz,dz)   

#heat transfer constants for rib outer wall segments 
F[0,7] = F_0(w_r,dz,o_t/4,o_t/4)     
F[1,7] = 2*F_0(dz,o_t/4,w_r,(w+w_r)/2)     
F[2,7] = F_0(w_r,o_t/4,dz,dz)     

#heat transfer constants for regen outer wall segments 
F[0,8] = F_0(w,dz,o_t/4,o_t/4)     
F[1,8] = 2*F_0(dz,o_t/4,w,(w+w_r)/2)     
F[2,8] = F_0(w,o_t/4,dz,dz)     



#creating coeficent matrix rows cols and vals (used to form the matrix) 
rows = []
cols = []
vals = []

#segment 1 
rows += np.full(4, 0).tolist()
cols += [0, 1, 4, 21]
vals += [1-F[0,0]-F[1,0]-conv[0], F[0,0], F[1,0], conv[0]]

#segment 2 
rows += np.full(4, 1).tolist()
cols += [1, 0, 2, 5]
vals += [1-2*F[0,0]-F[1,0], F[0,0], F[0,0], F[1,0]]

#segment 3 (etc...)
rows += np.full(4, 2).tolist()
cols += [2, 1, 3, 6]
vals += [1-2*F[0,0]-F[1,0], F[0,0], F[0,0], F[1,0]]

rows += np.full(4, 3).tolist()
cols += [3, 2, 7, 20]
vals += [1-F[0,0]-F[1,0]-conv[2], F[0,0], F[1,0], conv[2]]


rows += np.full(4, 4).tolist()
cols += [4, 5, 0, 21]
vals += [1-F[0,1]-F[1,1]-conv[1], F[0,1], F[1,1], conv[1]]

rows += np.full(4, 5).tolist()
cols += [5, 4, 6, 1]
vals += [1-2*F[0,1]-F[1,1], F[0,1], F[0,1], F[1,1]]

rows += np.full(4, 6).tolist()
cols += [6, 5, 7, 2]
vals += [1-2*F[0,1]-F[1,1], F[0,1], F[0,1], F[1,1]]

rows += np.full(4, 7).tolist()
cols += [7, 6, 8, 3]
vals += [1-F[0,1]-F[0,2]-F[1,1], F[0,1], F[0,2], F[1,1]]


rows += np.full(4, 8).tolist()
cols += [8, 7, 9, 20]
vals += [1-F[0,3]-F[0,4]-conv[3], F[0,3], F[0,4], conv[3]]

rows += np.full(4, 9).tolist()
cols += [9, 8, 10, 20]
vals += [1-2*F[0,4]-conv[3], F[0,4], F[0,4], conv[3]]

rows += np.full(4, 10).tolist()
cols += [10, 9, 11, 20]
vals += [1-2*F[0,4]-conv[3], F[0,4], F[0,4], conv[3]]

rows += np.full(4, 11).tolist()
cols += [11, 10, 12, 20]
vals += [1-F[0,4]-F[0,5]-conv[3], F[0,4], F[0,5], conv[3]]


rows += np.full(4, 12).tolist()
cols += [12, 11, 13, 16]
vals += [1-F[0,6]-F[0,7]-F[1,7], F[0,6], F[0,7], F[1,7]]

rows += np.full(4, 13).tolist()
cols += [13, 12, 14, 17]
vals += [1-2*F[0,7]-F[1,7], F[0,7], F[0,7], F[1,7]]

rows += np.full(4, 14).tolist()
cols += [14, 13, 15, 18]
vals += [1-2*F[0,7]-F[1,7], F[0,7], F[0,7], F[1,7]]

rows += np.full(3, 15).tolist()
cols += [15, 14, 19]
vals += [1-F[0,7]-F[1,7], F[0,7], F[1,7]]


rows += np.full(4, 16).tolist()
cols += [16, 17, 12, 20]
vals += [1-F[0,8]-F[1,8]-conv[4], F[0,8], F[1,8], conv[4]]

rows += np.full(4, 17).tolist()
cols += [17, 16, 18, 13]
vals += [1-2*F[0,8]-F[1,8], F[0,8], F[0,8], F[1,8]]

rows += np.full(4, 18).tolist()
cols += [18, 17, 19, 14]
vals += [1-2*F[0,8]-F[1,8], F[0,8], F[0,8], F[1,8]]

rows += np.full(3, 19).tolist()
cols += [19, 18, 15]
vals += [1-F[0,8]-F[1,8], F[0,8], F[1,8]]

rows += [20, 21]
cols += [20, 21]
vals += [1, 1]

#coefficent matrix
A = coo_matrix((vals, (rows, cols)), shape=(22, 22)).tocsr()

for i in range(round(duration/dt)):
    temps = A @ temps 

#stability check
diagonal = A.diagonal()[:20]
print(min(diagonal))
print(diagonal)


##Chat GPT plotting temps on a heat map
# ============================================================
# 2D TEMPERATURE HEAT MAP
# ============================================================

# Physical node temperatures only
T = temps[:20]

# Store polygons and corresponding temperatures
polygons = []
temperatures = []
node_numbers = []
node_centres = []


# Function to create rectangular thermal nodes
def add_node(node, x1, x2, y1, y2):

    polygon = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ]

    polygons.append(polygon)

    temperatures.append(T[node])

    node_numbers.append(node)

    node_centres.append((
        (x1 + x2)/2,
        (y1 + y2)/2
    ))


# ============================================================
# CHAMBER WALL
# ============================================================

# Nodes 0-3: chamber wall beneath cooling channel

for i in range(4):

    add_node(
        node=i,
        x1=0,
        x2=w,
        y1=i*t/4,
        y2=(i+1)*t/4
    )


# Nodes 4-7: chamber wall beneath rib

for i in range(4):

    add_node(
        node=i+4,
        x1=w,
        x2=w+w_r,
        y1=i*t/4,
        y2=(i+1)*t/4
    )


# ============================================================
# RIB
# ============================================================

# Nodes 8-11

for i in range(4):

    add_node(
        node=i+8,
        x1=w,
        x2=w+w_r,
        y1=t+i*h/4,
        y2=t+(i+1)*h/4
    )


# ============================================================
# OUTER WALL
# ============================================================

# Nodes 12-15: outer wall above rib

for i in range(4):

    add_node(
        node=i+12,
        x1=w,
        x2=w+w_r,
        y1=t+h+i*o_t/4,
        y2=t+h+(i+1)*o_t/4
    )


# Nodes 16-19: outer wall above cooling channel

for i in range(4):

    add_node(
        node=i+16,
        x1=0,
        x2=w,
        y1=t+h+i*o_t/4,
        y2=t+h+(i+1)*o_t/4
    )

# ============================================================
# COOLANT CHANNEL
# ============================================================

coolant_polygon = [
    (0, t),
    (w, t),
    (w, t+h),
    (0, t+h)
]

polygons.append(coolant_polygon)

temperatures.append(T_c)


# ============================================================
# CREATE HEAT MAP
# ============================================================

fig, ax = plt.subplots(figsize=(9, 8))

# Temperature colour scale
norm = Normalize(
    vmin=min(np.min(T), T_c),
    vmax=max(np.max(T), T_c)
)

# Create coloured polygons
collection = PolyCollection(
    polygons,
    array=np.array(temperatures),
    cmap="inferno",
    norm=norm,
    edgecolors="black",
    linewidths=0.5
)

ax.add_collection(collection)


# ============================================================
# NODE LABELS
# ============================================================

for node, (x, y) in zip(node_numbers, node_centres):

    ax.text(
        x,
        y,
        f"{T[node]:.0f}",
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        fontweight="bold"
    )

# ============================================================
# AXES AND FORMATTING
# ============================================================

ax.set_xlim(0, w+w_r)

ax.set_ylim(0, t+h+o_t)

# Convert axis display from metres to millimetres
from matplotlib.ticker import FuncFormatter

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x*1000:.1f}")
)

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, pos: f"{y*1000:.1f}")
)

ax.set_xlabel("Width (mm)")
ax.set_ylabel("Height (mm)")

ax.set_title(
    f"Chamber Wall Temperature Distribution\n"
    f"Time = {duration:.2f} s"
)

ax.set_aspect("equal")

# Colour bar
cbar = fig.colorbar(collection, ax=ax)

cbar.set_label("Temperature (K)")

plt.tight_layout()

plt.show()






