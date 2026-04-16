This dataset was collected on the biaxial device in the Rock Mechanics laboratory of the Penn State University by AM Eijsink. For details on the experimental procedure and roughness measurement procedure, please see the corresponding manuscript, currently submitted to EPSL. DOI to come.

The dataset consist of 5 experiments, 2 on gneiss: p5838 and p5848 and 3 on granitoid: p5897, p5905 and p5912. Roughness measurements were collected before the experiments and after the experiments, with the exception of after p5912. Roughness was measured with the Keyence optical profilometer in the Materials Characterization Lab at the Pennsylvania State University.

There are 3 types of data:
(1) raw mechanical and analyzed P-wave data matched in time. For raw acoustic data, contact the author (eijsink@psu.edu). There are 5 datasets for 5 experiments, submitted as .mat files with p####_datatable.mat where p#### denotes the experiment number as above. Units and meaning of variables explained below. 
(2) rate-and-state friction parameters as analyzed by the RSFit3000 MATLAB GUI (Skarbek and Savage, 2019). These data are named p####_RSFit3000.mat. For information about the data structure in these files, have a look at Skarbek and Savage (2019). 
(3) roughness data. These are .cag files that contain data as collected by the Keyence optical profilometer as well as the geometrical correction and analysis of the parameters. These files can be opened with the vk-X 3000 MultiFileAnalyzer software from Keyence. Point clouds as .csv files available upon request from corresponding author (eijsink@psu.edu). Files are labeled with the experiment number after which they were done, or when they were done before any experimenting. Most files include both sides of the faults, for the first two gneiss analysis the two sides of the fault ('shortside' and 'tall side') are in separate files.

Header explanation for (1) raw mechanical and analyzed P-wave data matched in time:
QpA_int		m3		the volume of water in pump A (upstream) 
QpB_int		m3		the volume of water in pump B (downstream)
effNS		MPa		effective normal stress
d_int		mm		shear/vertical displacement measured directly on the sample
mu		-		friction coefficient
compaction_ext	mm		normal/horizontal displacement measured at the piston (outside the pressure vessel)
compaction_int	mm		normal/horizontal displacement measured directly on the sample
sigmaN		MPa		normal stress
Pc_disp		mm		displacement of the hydraulic piston controlling confining fluid volume
Pc		MPa		confining pressure
PpA_disp_int	mm		displacement of the hydraulic piston controlling fluid volume in pump A (upstream)
PpA		MPa		fluid pressure in pump A (upstream)
PpB_disp_int	mm		displacement of the hydraulic piston controlling fluid volume in pump B (downstream)
PpB		MPa		fluid pressure in pump B (downstream)
Pp_average	MPa		average fluid pressure
d_ext		mm		shear/vertical displacement as measured at the piston (outside the pressure vessel)
tau		MPa		shear stress
sync		-		arbitrary pulses to link the time on the biax recording system to the acoustic data
time		s		time since start recording
v_int		um/s		shear displacement rate measured directly on the sample
v_ext		um/s		shear displacement rate measured at the piston (outside the pressure vessel)
timeshift	us		change in time of flight between all transmitter/receiver pairs
Amp		-		amplitude of P-wave (not calibrated, so arbitrary units)
RmsAmp		-		RMS amplitude of P-wave
maxAmp		-		maximum amplitude of P-wave
maxFreq		- 		Frequency of P-wave
avg_Amp				as above averaged for all transmitter/receiver pairs
avg_RmsAmp			as above averaged for all transmitter/receiver pairs
avg_timeshift			as above averaged for all transmitter/receiver pairs
v_acoustics	km/s		velocity of the P-wave for all transmitter/receiver pairs	
dP		Pa		pressure gradient along the fault
dhdt		mm/s		rate of fault-normal displacement, dilation/compaction rate
Q		m3/s		fluid flow rate along the fault (averaged for inflow/outflow pump A and B)
k		m2		permeability, uncorrected for storage
k_corr		m2		permeability, corrected for fluid storage due to opening of fault

QpA, QpB, VpA, VpB PpB_disp and PpA_disp with the extension _ext refer to additional measurements with an outside DCDT mounted to the hydraulic piston of the fluid pressure intensifiers. This was only done in p5838 and p5848 as it did not significantly improve data quality. 

