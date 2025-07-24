# TADR
MODVOLC_TADR_VOLUME
Access MODVOLC freely at http://modis.higp.hawaii.edu/
Select period of time, Volcano name, Aqua/Terra, day or night
Number of Alerts will show in Progress bar
Save Alert data using a unique name (e.g., Etna_2014)
Before running the code edit the following: 
in line 257 and 258 add a path do data you downloaded from MODVOLC
in line 260 chose 'model1' for 100'C lava temperature or 'model2' for 500'C lava temperature
Press 'run' the code will go through these steps
STEP 1. Atmospheric Correction
STEP 2. Fraction of Lava at 'lower' (100'C) or 'upper' (500'C) modelled temperature in MODIS pixel
STEP 3. Correction of MODIS pixel size (nadir = 1000000 i.e. 1 square km) using 'SatZen' angle
STEP 4. Area of lava (100'C and 500'C)
STEP 5. Calculate Mtot Heat Loss in W (Harris et al., 1997) or Harris, 2013 'Thermal Remote Sensing of Active Volcanoes'
STEP 6. Calculate TADR (Harris et al., 1997)
csv file will be written for TADR, summed TADR and peaks
STEP 7. Calculate Volume (Trapesoidal Integration)
RESULTS: Plot will show curves for TADR (y-axis in cubic meters per second and in x-axis time in days)
RESULTS: Console will show at the bottom line TADR original peaks (e.g., 7.487196e+07) in cubic meters
         Code can also 'smooth' the curves (read TADR (smoothed peaks) value (e.g., 8.505223e+07) in cubic meters
NOTE 1: It is expected for 'smoothed TADR peaks' approach to produce higher erupted volume values.
NOTE 2: Compositional input parameters and background temperatures are target dependent and must be adjusted accordingly.
For example, compositional parameters/background temps for Mt. Etna are different from Nyamuragira and must be adjusted accordingly. 
////////////
