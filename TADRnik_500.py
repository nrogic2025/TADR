# -*- coding: utf-8 -*-
"""
Spyder Editor

TADR computation.
"""

# Import all packages
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

# Define constants (UPPERCASE) - these never change (global variables)
SIGMA = 5.67E-08 # [Wm-2.K-4] Stefan Boltzmann constant
GRAV = 9.807   # [m.s-2] gravity
MODIS_AREA_NADIR = 1E6  # [m2] MODIS pixel area at nadir



# Define functions
def atmospheric_correction(radiance, trans, emiss, upwelling_radiance):
    """
    Correct radiance for atmosphere.
    
    Uses atmospheric correction from MODTRAN or Barsy for date/time/location.

    Parameters
    ----------
    radiance : np.array
        MODIS radiance band / wavelength... (units: []).
    trans : float
        MODTRAN or Barsy.
    emiss : float
        measured for material.
    L_upwelling : float
        MODTRAN or Barsy.

    Returns
    -------
    corrected_radiance: np.array
    """
    
    corrected_radiance = (radiance / trans) * emiss - upwelling_radiance 
    return corrected_radiance


def Fraction_lava(rad_corrected,rad_bkg,rad_lava):
    """
    Calculate fraction of lava at temperature in MODIS pixel
    
    Parameters
    ----------
    corrected_radiance : np.array
        DESCRIPTION.
    bkg_radiance : float
        DESCRIPTION.
    lava_radiance : float
        DESCRIPTION.

    Returns
    -------
    fraction_lava: np.array.

    """

    Fraction_lava = (rad_corrected-rad_bkg)/(rad_lava-rad_bkg)
    return Fraction_lava
    
def Corrected_Pixel_area(satellite_zenith):
    """
    Adjust the MODIS pixel size based on SatZen angle

    Parameters
    ----------
    pixel_area : float
        1000000 m at nadir.
    satellite_zenith : np.array
        per pixel Satellite Zenith angle.

    Returns
    -------
    corr_pixel_area: np.array.

    """
    
    corr_pixel_area = MODIS_AREA_NADIR/np.cos(np.radians(satellite_zenith))
    return corr_pixel_area

def Area_lava(Fraction_lava):
    """
    Calculate area of lava at 100 K within MODIS pixel

    Parameters
    ----------
    fraction_lava : np.array
        DESCRIPTION.
    pixel_area : np.array
        DESCRIPTION.

    Returns
    -------
    area_lava.

    """
    
    Area_lava = Fraction_lava*MODIS_AREA_NADIR
    return Area_lava

def Heat_Loss_M_rad(Area_lava,emiss,temp_lava):
    """
    Calculate radiative heat loss (M_rad)

    Parameters
    ----------
    Area_lava : TYPE
        DESCRIPTION.
    emiss : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    Temperature_lava : TYPE
        DESCRIPTION.

    Returns
    -------
    M_rad: np.array.

    """
    M_rad=Area_lava*emiss*SIGMA*pow(temp_lava, 4)
    return M_rad

def Heat_Loss_M_conv(Area_lava,k,alpha,rho_air,mu,kappa,temp_lava,temp_bkg):
    """
    

    Parameters
    ----------
    Area_TOT : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    rho_air : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    kappa : TYPE
        DESCRIPTION.
    T_lava : TYPE
        DESCRIPTION.
    T_BKG : TYPE
        DESCRIPTION.

    Returns
    -------
    M_conv: np.array.

    """
    M_conv=0.14*Area_lava*k*(GRAV*alpha*rho_air/mu*kappa)**0.3333*(temp_lava-temp_bkg)**1.3333
    return M_conv
    
def Total_Heat_Loss(M_rad, M_conv):
    """
    

    Parameters
    ----------
    M_rad+M_conv : TYPE
        DESCRIPTION.

    Returns
    -------
    M_TOT: np.array.

    """
    M_TOT=(M_rad + M_conv)
    return M_TOT
    
def TADR(M_TOT,dre_rho,dre_cp,delta_temp,cl,cryst):
    """
    Calculate time averaged discharge rate

    Parameters
    ----------
    M_TOT : TYPE
        DESCRIPTION.
    DRE_rho : TYPE
        DESCRIPTION.
    DRE_Cp : TYPE
        DESCRIPTION.
    delta_T : TYPE
        DESCRIPTION.
    CL : TYPE
        DESCRIPTION.
    cryst : TYPE
        DESCRIPTION.

    Returns
    -------
    TADR: np.array.

    """
    TADR=M_TOT/(dre_rho*((dre_cp*delta_temp)+(cl*cryst)))
    return TADR
#def other_funcs():
    pass

#def other_funcs():
    #pass

def tadr():
    """
    Calculate time-avereaged discharge rate.

    Returns
    -------
    None.

    """
    # Call all the other functions above to get to tadr
    tadr = 0 # <-replace with harris equation
    return tadr


def read_modvolc(input_file):
    """
    Return a DataFrame of MODVOLC values from a text file.

    Parameters
    ----------
    input_file : TYPE
        DESCRIPTION.

    Returns
    -------
    df: pd.DataFrame

    """
    df = pd.read_csv(input_file, delim_whitespace=True)
    df.loc[:, 'datetime'] = pd.to_datetime(df.loc[:, "UNIX_Time"], unit="s")
    return df

def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window

#def main():
"""Run code on input files"""

# Ask user for input file

# Local variables (lower_case) - these we may want to change each run
data_path = r"C:\Users\Nic\Desktop\MODVOLC\TARGETS\\"
modvolc_file = data_path + r"\modis_alert_data (2024).txt"

model = 'model2'  # or 'model2'

if model == 'model1':
    temp_lava_c = 100
    temp_lava = 373  # [K] model_100 lava temperature
    rad_lava = 21.6   # [Wm-2.sr-1.micron-1] radiance
    k = 2.78E-02 # [W.m-1.K1] thermal conductivity
    alpha = 3.09E-03    # [K-1] cubic expansivity
    rho_air = 1.0928    # [kg.m-3] air density
    mu = 1.96E-05   # [Pa.s] dynamic viscosity
    kappa = 2.50E-2   # [m2.s-1] thermal diffusivity 
elif model == 'model2':
    temp_lava_c = 500    
    temp_lava = 773  # [K] model_100 lava temperature
    rad_lava = 157.5   # [Wm-2.sr-1.micron-1] radiance
    k = 3.95E-02 # [W.m-1.K1] thermal conductivity
    alpha = 2.0E-03    # [K-1] cubic expansivity
    rho_air = 0.706    # [kg.m-3] air density
    mu = 2.68E-05   # [Pa.s] dynamic viscosity
    kappa = 5.40E-2   # [m2.s-1] thermal diffusivity
    
#temp_bkg = 276  # [K] background temperature
temp_bkgs = np.array([260.07, 259.33, 259.54, 263.13, 264.89, 269.72, 270.85, 269.75, 268.40, 269.36, 260.71, 260.76])
#rad_bkg = 6   # [Wm-2.sr-1.micron-1] radiance
rad_bkgs = np.array([4.63, 4.56, 4.58, 4.91, 5.07, 5.55, 5.66, 5.55, 5.42, 5.51, 4.68, 4.68])
emiss = 0.95    # [unitless] emissivity
trans = 0.96    # [%] atmospheric transmissivity
dre_rho = 2600  # [kg.m-3] dense rock equivalent density
dre_cp = 1150   # [J.kg-1.K-1] DRE specific heat capacity
delta_temp = 150   # [K] T difference between erupt_T and stop_T
cryst = 0.45  # [%] Fraction of crystals
cl = 2.90E+05    # [J.kg-1] latent heat of crystallization
hc = 10  # [W.m-2.K-1] convective heat transfer coefficient 
u_rad = 0.14   #  [Wm-2.sr-1.micron-1] upwelling radiance

X = 2.15802E-06   # TADR_100 coefficient
# satellite_zenith = 30 #angle


# Read in data
df = read_modvolc(modvolc_file)
print(df.head(10))

# Clean the data
# TODO: Read in Etna background temp (radiance), exclude where rad < bg_rad
df['rad_bkg'] = rad_bkgs[df['Mo']-1]  # [Wm-2.sr-1.micron-1] radiance
df['rad_corrected'] = atmospheric_correction(df.B31, trans, emiss, u_rad)
print("radiance_corrected", df.rad_corrected)

# Find where atmospherically corrected rad < background
print(np.where((df.rad_corrected <= df.rad_bkg)))  # <- show the rows we are about to kick out (note <=)
df = df[df.rad_corrected > df.rad_bkg]  # Only keep rows where rad_corr is larger (note >)

f_lava = Fraction_lava(df.rad_corrected, df.rad_bkg, rad_lava)
print("fraction_of_lava", f_lava)

a_pixel = Corrected_Pixel_area(df.SatZen)
print("corrected_pixel_area", a_pixel)

a_lava = Area_lava(f_lava) 
print("area_of_lava", a_lava)

m_rad = Heat_Loss_M_rad(a_lava,emiss,temp_lava)
print("Heat_Loss_M_rad", m_rad)

df['temp_bkg']=temp_bkgs[df['Mo']-1]
m_conv = Heat_Loss_M_conv(a_lava, k, alpha, rho_air, mu, kappa, temp_lava, df.temp_bkg)
print("Heat_Loss_M_conv", m_conv)

m_tot = Total_Heat_Loss(m_rad, m_conv)
print("Total_Heat_Loss", m_tot)    

tadr = TADR(m_tot, dre_rho, dre_cp, delta_temp, cl, cryst)
print("TADR:", tadr)

# Add tadr column
df.loc[:, "tadr"] = tadr
df = df.sort_values('datetime')

# Filter out negative tadr
# df2 = df[df.loc[:, "tadr"] > 0]

# Plot tadr vs date
fig, ax = plt.subplots(figsize=(15, 5))
df.plot('datetime', 'tadr', 'scatter', marker='o', ax=ax)

# Find timegaps and plot raw data points
df['timegap_gt_7days'] = df['datetime'].diff() > pd.to_timedelta('7 days')
df[df.timegap_gt_7days].plot('datetime', 'tadr', 'scatter', marker='x', color='r', ax=ax)
fig.autofmt_xdate()
ax.set_xbound([datetime.date(2001, 6, 1), datetime.date(2001,8, 8)])
ax.set_ylim(0, 2)

# Save 7 day time gaps to file
df[df.timegap_gt_7days][['Year', 'Mo', 'Dy', 'datetime']].to_csv(data_path + r"\new_eruption_dates.csv")

df.to_csv(data_path + r"\TADR.csv")  # Save TADR to file
summed_tadr = df.groupby('datetime')['tadr'].sum()
# add summed_tadr column
df.loc[:, "summed_tadr"] = summed_tadr
print(summed_tadr)


# Use find peaks to automatically get indices of the peaks
#peaks, _ = find_peaks(summed_tadr, prominence=1)  # sometimes overestimates area because of outliers
peaks, _ = find_peaks(summed_tadr, height=0)  # min height 0 is every increase in TADR (tighter fit to curve)
# Add back in the first and last index
peaks = np.array([0, *peaks, -1])

#Use manual peak indices instead
#peaks = np.array([0, 1, 2, 3, 50, 87, 280, 380, 420, 490, 550, 600, -1])


smoothed_peaks = moving_average(summed_tadr[peaks], 3)
smoothed_peaks = np.array([summed_tadr[0], *smoothed_peaks, summed_tadr[-1]])

# For each point, take the higher of original TADR and smoothed TADR
smoothed_peak_maximums = np.max([summed_tadr[peaks], smoothed_peaks], axis=0)

datetimes = summed_tadr.index.to_pydatetime()  # datetime objects
time_since_t0 = (datetimes - datetimes[0]) # timedelta objects
seconds = np.array([dt.total_seconds() for dt in time_since_t0])  # [s]

y = summed_tadr[peaks]
x = seconds[peaks]
vol_lava = np.trapz(y, x)
print(f'TADR (peaks height=0): {vol_lava:e}')

y2 = smoothed_peak_maximums
vol_lava_smoothed = np.trapz(y2, x)
print(f'TADR (smoothed peaks): {vol_lava_smoothed:e}')

fig, ax = plt.subplots()
x_indices = np.arange(len(summed_tadr))
ax.plot(x_indices, summed_tadr, 'bx')
ax.plot(x_indices[peaks], summed_tadr[peaks], "b", label='original')
#ax.plot(x_indices[peaks], smoothed_peaks, "r--", label='moving avg')
ax.plot(x_indices[peaks], smoothed_peak_maximums, "g", label='smoothed max')
ax.legend()

datetimes_str = np.array([dt.strftime("%d/%m/%Y") for dt in datetimes])
out = np.array([datetimes_str[peaks], smoothed_peak_maximums]).T
np.savetxt(data_path + r"\peaks_smoothed.csv", out, delimiter=",", fmt="%s")

#ax.plot(summed_tadr.index, summed_tadr, 'xb')
#ax.plot(summed_tadr.index[peaks], summed_tadr[peaks], "g")


#summed_tadr.plot(color='blue')
summed_tadr.to_csv(data_path + r"\summed_TADR.csv")

#tadr_tot = print(sum(tadr))
# Run TADR calculation - call functions on data we read in and our 
#  defined variables for this run above, return the TADR

# m_tot = total_heat_loss(param1, param2)
    # print(time_avg_dis_rate(m_tot, params...)
    
#if __name__ == '__main__':
#    main()
    