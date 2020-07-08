import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import xlrd
from pylab import *
import requests
import io

wdir = "C:/Users/40kmp/Desktop/internship/01_data"
dfStates = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/T_vwc_all2.csv")
dfLayers = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/ALT_WT_strati_all.csv")  
dfLayers2 = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/ALT_WT_strati_Apos.csv")
dfThermProp = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/K_C_A.csv")
sns.set(font_scale=1)
df = pd.DataFrame(data = dfLayers, columns = ['Frost table depth, cm','Water table depth, cm','Moss layer depth, cm','Peat layer depth, cm','Snow Depth, cm'])
flatui = ["#D3CB5D", "#3498db", "#44CD27", "#FF7F00", "#C5C5C5"]  #pallete colors
sns.set_palette(flatui)
sns.palplot(sns.color_palette())

#%%
#BOXPLOTS

dfStates = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/T_vwc_all2.csv")
dfLayers = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/ALT_WT_strati_A.csv")  
dfLayersC = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/ALT_WT_strati_C.csv")
dfLayersD = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/ALT_WT_strati_D.csv")
dfThermProp = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/K_C_A.csv")
sns.set(font_scale=1)

df = pd.DataFrame(data = dfLayers, columns = ['Frost table depth, cm','Water table depth, cm','Moss layer depth, cm','Peat layer depth, cm','Snow Depth, cm'])
sns.set_style(style='whitegrid')

#%%
#SITE A

flatui = ["#A569BD", "#5DADE2", "#45B39D", "#F5B041", "#BFC9CA"]
df = pd.DataFrame(data = dfLayers, columns = ['Frost table depth, cm','Water table depth, cm','Moss layer depth, cm','Peat layer depth, cm','Snow Depth, cm'])
fig, ax = plt.subplots(figsize=(10,10))

xx = sns.boxplot(x="variable", y="value", data=pd.melt(df),palette=flatui, ax= ax)

d = ax.axhline(y=0, xmin=0.0, xmax=1.0, color='k',label='Surface level',linewidth=3)
first_legend = ax.legend(handles=[d], loc='upper left')

ax.set_ylabel('Depth, cm')
ax.set_xlabel(' ')
ax.set_title('Site A')
plt.xticks(rotation=90)

#plt.xlim(-1, 10)
#plt.ylim(-55, 50)

plt.savefig('boxplotA.png', dpi=300, bbox_inches = "tight")
plt.show()
#%%
#SITE C

flatui = ["#A569BD", "#5DADE2", "#45B39D", "#F5B041", "#BFC9CA"]
df = pd.DataFrame(data = dfLayersC, columns = ['Frost table depth, cm','Water table depth, cm','Moss layer depth, cm','Peat layer depth, cm','Snow Depth, cm'])
fig, ax = plt.subplots(figsize=(10,10))

xx = sns.boxplot(x="variable", y="value", data=pd.melt(df),palette=flatui, ax= ax)

d = ax.axhline(y=0, xmin=0.0, xmax=1.0, color='k',linewidth=3,label='Surface level',)
first_legend = ax.legend(handles=[d], loc='upper left')

ax.set_ylabel('Depth, cm')
ax.set_xlabel(' ')
ax.set_title('Site C')
plt.xticks(rotation=90)

#plt.xlim(-1, 10)
#plt.ylim(-55, 50)

plt.savefig('boxplotC.png', dpi=300, bbox_inches = "tight")
plt.show()


#%%
#SITE D

flatui = ["#A569BD", "#5DADE2", "#45B39D", "#F5B041", "#BFC9CA"]
df = pd.DataFrame(data = dfLayersD, columns = ['Frost table depth, cm','Water table depth, cm','Moss layer depth, cm','Peat layer depth, cm','Snow Depth, cm'])
fig, ax = plt.subplots(figsize=(10,10))

xx = sns.boxplot(x="variable", y="value", data=pd.melt(df),palette=flatui, ax= ax)

d = ax.axhline(y=0, xmin=0.0, xmax=1.0, color='k',linewidth=3,label='Surface level')
first_legend = ax.legend(handles=[d], loc='upper left')

ax.set_ylabel('Depth, cm'), ax.set_xlabel(' '), ax.set_title('Site D')
plt.xticks(rotation=90)

#plt.xlim(-1, 10)
#plt.ylim(-55, 50)

plt.savefig('boxplotD.png', dpi=300, bbox_inches = "tight")
plt.show()

#%%
#SNOW DEPTH

sns.set_style(style='whitegrid')
plt.rcParams.update({'font.size': 22})
dfLayersSN = pd.read_csv("SnowDepthBoxplot.csv") 
flatui = ["#A9F0FF", "#52DAF6", "#1BA7C4","#026276"]  #pallete colors
df = pd.DataFrame(data = dfLayersSN, columns = ['Site A, cm','Site B, cm','Site C, cm','Site D, cm'])
fig, ax = plt.subplots(figsize=(10,10))

xx = sns.boxplot(x="variable", y="value", data=pd.melt(df),palette=flatui, ax= ax)
#xx = sns.swarmplot(x="variable", y="value", data=pd.melt(df), color="red")
ax.set_ylabel('Depth, cm')
ax.set_xlabel(' ')
plt.ylim(0, 250)
plt.savefig('Snow Depth.png', dpi=300, bbox_inches = "tight")
plt.show()


##############################################################################################

#%%
#Calendar heat map
b1=pd.read_csv('999_BA00000048DF4B41_092719.csv')

b1['Date/Time'] = pd.to_datetime(b1['Date/Time'])

b1 = b1.set_index('Date/Time')

b1_mean = b1.resample('D').mean()

df = b1_mean
#%%
df.rename(columns = {'Value' : 'Snow'}, inplace = True)
df.loc[df.Snow < 0, 'mask'] = "Lower"
df.loc[df.Snow > 0, 'mask'] = "Higher"

#%%

from matplotlib import colors
value_to_int = {j:i+1 for i,j in enumerate(pd.unique(df['mask'].ravel()))}
df = df.replace(value_to_int)
cal = {'2018': df[df.index.year == 2018], '2019': df[df.index.year == 2019]}

# Define Ticks
DAYS = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(2, 1, figsize = (20,6))
for i, val in enumerate(['2018', '2019']):
    start = cal.get(val).index.min()
    end = cal.get(val).index.max()
    start_sun = start - np.timedelta64((start.dayofweek) % 7, 'D')
    end_sun =  end + np.timedelta64(7 - end.dayofweek, 'D')

    num_weeks = (end_sun - start_sun).days // 7
    heatmap = np.full([7, num_weeks], np.nan)
    ticks = {}
    y = np.arange(8) - 0.5
    x = np.arange(num_weeks + 1) - 0.5
    for week in range(num_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 1:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = cal.get(val).loc[date, 'mask']

    cmap = colors.ListedColormap(['tomato', 'skyblue'])
    mesh = ax[i].pcolormesh(x, y, heatmap, cmap = cmap, edgecolors = 'grey')

    ax[i].invert_yaxis()
        # Hatch for out of bound values in a year
    ax[i].patch.set(hatch='xx', edgecolor='black')

    # Set the ticks.
    ax[i].set_xticks(list(ticks.keys()))
    ax[i].set_xticklabels(list(ticks.values()))
    ax[i].set_yticks(np.arange(7))
    ax[i].set_yticklabels(DAYS)
    ax[i].set_ylim(6.5,-0.5)
    ax[i].set_aspect('equal')
    ax[i].set_title(val, fontsize = 15)

# Add color bar at the bottom
cbar_ax = fig.add_axes([0.25, -0.10, 0.5, 0.05])
fig.colorbar(mesh, orientation="horizontal", pad=0.2, cax = cbar_ax)
n = len(value_to_int)
colorbar = ax[1].collections[0].colorbar
r = colorbar.vmax - colorbar.vmin
colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
colorbar.set_ticklabels(list(value_to_int.keys()))    
fig.suptitle('Zero degree Celsius', fontweight = 'bold', fontsize = 25)
fig.subplots_adjust(hspace = 0.5)
fig.savefig( 'zero.png', bbox_inches='tight') 

#%%

from matplotlib import colors

# Turn data frame to a dictionary for easy access
cal = {'2018': df[df.index.year == 2018], '2019': df[df.index.year == 2019]}

# Define Ticks
DAYS = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat','Sun']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(2, 1, figsize = (20,6))
for i, val in enumerate(['2018', '2019']):
    start = cal.get(val).index.min()
    end = cal.get(val).index.max()
    start_sun = start - np.timedelta64((start.dayofweek) % 7, 'D')
    end_sun =  end + np.timedelta64(7 - end.dayofweek, 'D')

    num_weeks = (end_sun - start_sun).days // 7
    heatmap = np.full([7, num_weeks], np.nan)    
    ticks = {}
    y = np.arange(8) - 0.5
    x = np.arange(num_weeks + 1) - 0.5
    for week in range(num_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 1:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = cal.get(val).loc[date, 'Snow']
    mesh = ax[i].pcolormesh(x, y, heatmap, cmap = 'rainbow', edgecolors = 'grey')

    ax[i].invert_yaxis()

    # Set the ticks.
    ax[i].set_xticks(list(ticks.keys()))
    ax[i].set_xticklabels(list(ticks.values()))
    ax[i].set_yticks(np.arange(7))
    ax[i].set_yticklabels(DAYS)
    ax[i].set_ylim(6.5,-0.5)
    ax[i].set_aspect('equal')
    ax[i].set_title(val, fontsize = 15)

    # Hatch for out of bound values in a year
    ax[i].patch.set(hatch='xx', edgecolor='black')

# Add color bar at the bottom
cbar_ax = fig.add_axes([0.25, -0.10, 0.5, 0.05])
fig.colorbar(mesh, orientation="horizontal", pad=0.2, cax = cbar_ax)
colorbar = ax[1].collections[0].colorbar
r = colorbar.vmax - colorbar.vmin
fig.suptitle('Temperature', fontweight = 'bold', fontsize = 25)
fig.subplots_adjust(hspace = 0.5)
fig.savefig( 'temperature.png', bbox_inches='tight') 

############################################################################################

#%%
#SEDIMENT LAYERS

xls = xlrd.open_workbook(r'MW_Lena17_frozen.xlsx', on_demand=True)
sns.set_style(style='ticks')

l = xls.sheet_names()
sdm = []
for i in range(len(l)):
    sediment = pd.read_excel('MW_Lena17_frozen.xlsx', header=22, sheet_name=l[i])
    sdm.append(sediment)
for i in range(len(sdm)):
    plt.plot(sdm[i]['Size,μm'], sdm[i]['Volume,%'],color='r',linewidth=0.8)

#plt.xticks(rotation=90)
plt.xscale('log')
#plt.title('Grain size distribution pattern of Sediment, logarithmic scale')
plt.xlabel('Size,LG')
plt.ylabel('Volume,%')
#plt.gca().legend(('Frozen sediment layer'))
plt.savefig('sediment_log.png', dpi=300, bbox_inches='tight')


#%%
xls2 = xlrd.open_workbook(r'MW_Lena17_peat.xlsx', on_demand=True)
sns.set_style(style='ticks')

l2 = xls2.sheet_names()
sdm2 = []
for i in range(len(l2)):
    sediment2 = pd.read_excel('MW_Lena17_peat.xlsx', header=22, sheet_name=l2[i])
    sdm2.append(sediment2)
for i in range(len(sdm2)):
    plt.plot(sdm2[i]['Size,μm'], sdm2[i]['Volume,%'],'b',linewidth=0.8)
plt.legend()
#plt.xticks(rotation=90)
plt.xscale('log')
#plt.title('Grain size distribution pattern of Sediment, logarithmic scale')
plt.xlabel('Size,LG')
plt.ylabel('Volume,%')

#plt.gca().legend(('Upper peat layer'))
plt.savefig('sediment_log.png', dpi=300, bbox_inches='tight')
#%%
xls3 = xlrd.open_workbook(r'MW_Lena17_mineral.xlsx', on_demand=True)
sns.set_style(style='ticks')

l3 = xls3.sheet_names()
sdm3 = []
for i in range(len(l3)):
    sediment3 = pd.read_excel('MW_Lena17_mineral.xlsx', header=22, sheet_name=l3[i])
    sdm3.append(sediment3)
for i in range(len(sdm3)):
    plt.plot(sdm3[i]['Size,μm'], sdm3[i]['Volume,%'],color='g',linewidth=1.2)

#plt.xticks(rotation=90)
plt.xscale('log')
#plt.title('Grain size distribution pattern of Sediment, logarithmic scale')
plt.xlabel('Size,LG')
plt.ylabel('Volume,%')
#plt.gca().legend(('Mineral layer'))
plt.savefig('sediment_log.png', dpi=300, bbox_inches='tight')

###########################################################################################

#%%

#Correlation heatmap
alles = pd.read_csv('alles2.csv')
alles.head(10)
sns.heatmap(alles.corr(), annot = True, vmin=-1, vmax=1, center= 0,fmt='.1g',cmap='rainbow',linewidths=1, linecolor='black',)
plt.savefig('heatmap.png', dpi=300, bbox_inches = "tight")

#Scatter plot to check Correlation heatmap
plt.scatter(x=alles['Humus_mineral'],y=alles['vwc'])

############################################################################################

#%% 
#Comparison of soil constituents

xx =pd.read_csv('K_C_A.csv')
xx.head(10)

sns.pairplot(data=xx,kind='scatter',diag_kind='kde')
plt.tight_layout()
plt.savefig('t-vwc_hs-k-c.png', dpi=300, bbox_inches='tight')

#############################################################################################
#%%

#Comparison of thermal properties
dfStates2 = pd.read_csv("C:/Users/40kmp/Desktop/internship/01_data/T_vwc_all2.csv")
sns.set_style(style='ticks')
cols = ['Thermal_conductivity,(W/(m*K))', 'Heat_capacity,J/K','Volumetric_water_content,%']
hue= 'layer' 
hue_order= ['moss', 'humus', 'mineral']
colors = ['red','lawngreen', 'm']

dfPlot = dfStates2
pStates = sns.pairplot(data=dfPlot, vars=cols, hue=hue, hue_order=hue_order,
                 dropna=False, palette=colors,kind='scatter')#, diag_kind='kde')

axes = pStates.axes
axes[2,1].set_xlim(0,4)

axes[0,1].set_ylim(0,1.5), axes[0,2].set_ylim(0,1.5)
axes[1,0].set_ylim(0,5), axes[1,2].set_ylim(0,5)
axes[2,0].set_ylim(0,90), axes[2,1].set_ylim(0,90)

plt.savefig('K-C-VWC.png', dpi=300, bbox_inches = "tight") 