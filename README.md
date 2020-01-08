###[Yasuki Tachibana]
# Numerical Code to Solve Relativistic Ideal Hydrodynamic Equation by Piecewise Parabolic Method (PPM) in $\tau$-$\eta_{\rm s}$ Coordinates with Discretized Christoffel Symbols 


## Preparation

To dowonload data file for Equatin of State, 
move to ```PPM``` and run ```getEOSforPPM.sh```:

```bash
	cd YOUR_CODE_LOCATION/src/hydro/PPM
	./get_eos_for_ppm.sh
	cd YOUR_CODE_LOCATION
```

## Compile and Run

With cmake:

```bash
	mkdir build
	cd build
	cmake ..
	make -j
```

If you do not have any problem in compilation,
you can find example executable file ```PPM_ADS``` for ideal hydro simulation and xml file ```source_ads_test.xml``` for its setting.
Those are generated from ```PPM_ADS.cc``` and ```source_ads_test.xml``` 
in ```example``` directory. 
You can run the code for one event by typing,

```bash
	./PPM_ADS source_ads_test.xml 1
```

Once finishing the run, you can find output files, 
```ppm_profile_run_0.txt``` for evolving hydro profile in binary and ```ppm_hypersurface_run_0.txt``` for freezeout hypersurface information in txt. 

## NOTES
Although the hydro codes are embedded in JETSCAPE, currently there is no communication between jet evolution and hydro evolution.
Also it should be noted that the codes to generate source term belongs to the hydro module ```PPM``` and are NOT considered as independent module here.

## Source codes for hydro
You can find ```PPM.cc``` and ```PPM.h``` in ```src/hydro``` and src codes associating them in ```src/hydro/PPM```. Those are the codes for the ideal hydro used in ```PPM_ADS.cc```
Blief explanations for some of them are below:

1. ```PPM/PPMprinter.cc```
The code to generate the output file storing the evolving hydro profile.

1. ```PPM/PPMfreezeout.c```
The code to generate the output file storing the freezeout surface information.

1. ```PPM/PPMpartonAdS.cc```
The code to load information of partons depositing energy and momentum into the medium fluid.

1. ```PPM/PPMpartonAdS.cc```
The code to give gaussian profile for the source generated for partons loaded in "PPM/PPMpartonAdS.cc".

1. ```PPM/PPMinitial.cc```
The code to generate initial condition of hydro profile. 

## XML file and Settings
Since the structure is stolen from JETSCAPE, it is the same to JETCAPE codes essentially. 

### XML File: ```source_ads_test.xml```

#### 1. ```<name>PPM</name>```
```<taus>0.6</taus>```: the starting proper time of hydro calculation in tau-eta coordinates in [fm/c]. In Cartesian coordinates, the starting time in the lab is always set to zero. 

```<store_info>0</store_info>```: the flag to turn off and on the storing the hydro profile info. It is still under construction. Do not do anything now.

```<s_factor>57.00</s_factor>```: the overall normalization factor for the initial condition generated by Trento(+Freestreaming).

```<whichEOS>1</whichEOS>```: the flag to specify the equation of state for hydro calculation. (0: Ideal Massless QGP gass, 1: Lattice QCD [https://doi.org/10.1016/j.physletb.2014.01.007] )

```<EOSfiles>../src/hydro/PPM/EOS</EOSfiles>```: the path of the directory containing the tables of Lattice QCD EoS.

```<profileType>4</profileType>```: the flag to specify the initial condition and coordinate system for hydro calculation. (0:Trento+Freestreaming [tau-eta] underconstruction, 1: Bjorken [tau-eta], 2:DynamicalBrick [t-z], 3:3D Gaussian[t-z] 4:Transverse Input (from Dani) [tau-eta])

```<addCell>1</addCell>```: the flag to add fluid cells to those used in Trento(+Freestreaming). 

```<T0>0.25</T0>```: the temperature at the center for Bjorken, Brick, and 3D Gaussian initial conditions in [GeV].

```<nt>60</nt>```: the maximum number of  (proper) time steps in hydro calculation. For the case of isothermal freezeout, hydro calculation can stop before the last proper time step specified here. 

```<dtau>0.3</dtau>```: the size of the (proper) time step in [fm/c].

```<nx>193</nx>```: the number of the grids in x.

```<ny>193</ny>```: the number of the grids in y. 

```<neta>95</neta>```: the number of the grids in eta (or z). 

```<dx>0.3</dx>```: the size of the grid in x and y in [fm].

```<deta>0.3</deta>```: the size of the grid in eta (or z in [fm]).

```<source>2</source>```: the frag to specify the source term for energy-momentum deposition by jet. (0:No source, 1:Causal Diff ( Abailable only for Cartesian coordinates Now), 2:Gaussian)

```<profileInput>../hydro_profile/Smooth_initial.dat</profileInput>```: the path of the initial profile file used when ```profileType``` is set to 4.

```<initProfileLong>1</initProfileLong>```: the flag to specify the longitudinal profile when ```profileType``` is set to 4. (0: Bjorken, 1: Flat+Gauss[w/ parameters for 5.02 TeV] )

```<writeOutput>1</writeOutput>```: the flag to turn of saving the evolving hydro profile in binary
        
```<profileOutput>ppm_profile</profileOutput>```: the path/header of filename for the evolving hydro profile in binary. 

```<freezeout>1</freezeout>```: the flag to specify the freezeout. (0:No Freezeout, 1:Isothermal, 2, Isochronous)

```<T_freezeout>0.14</T_freezeout>```: the freezeout temperature for the case of isothermal freezeout (```freezeout``` is set to 1).
        
```<surface_name>ppm_hypersurface</surface_name>```: the path/header of filename for the freezeout surface information in txt. 
       
```<rapidity_window>3.5</rapidity_window>```: the maximum absolute value of space-time rapidity in the output files for the evolving hydro profile and the freezeout surface information. The area with space-time rapidity larger than this value do not show up in any output files. It is introduced to reduce the output file sizes.

```<transverse_square>12</transverse_square>```: the maximum absolute value of x and y in [fm] in the output files for the evolving hydro profile.
        
        
#### 1. ```<name>PPMsourceGauss</name>```
```<sourceInput>../source_list</sourceInput>```: the path of the directory containing the tables of sources. The file for run X must have the filename ```source_run_X.txt```. 

```<tau_thermal>0.0</tau_thermal>```: the (proper) time delay for the source injection in [fm/c]. 
          
```<sigma_trans>0.4</sigma_trans>```: the gaussian width of the source profile in x and y directions in [fm]. 

```<sigma_long>0.4</sigma_long>```: the gaussian width of the source profile in eta direction. 


## ```PPMpartonAdS``` and files in ```source_list```

The code ```PPM/PPMpartonAdS.cc``` reads the file of source (parton) information in ```source_list```.

In the curren set-up the structure of the file is 
```#id, pid, status, tau, x, y, eta_s, e, px, py, pz, de, dpx, dpy, dpz [in fm and GeV]```.

The explanation for each element is, 

```id```: just a serial number of the core of the source (parton). It is not used in the current code setting.

```pid```: pid of the core of the source (parton). It is not used in the current code setting.

```status```: status of the core of the source (parton). If ```status``` is set to -1, it is considered as a negative parton. 

```tau```: proper time of the core of the source (parton) in tau-eta coordinates in [fm]. 

```x```: the x-component of the location of the core of the source (parton) in [fm]. 

```y```: the y-component of the location of the core of the source (parton) in [fm]. 

```eta_s```: the space-time rapidity-component of the location of the core of the source (parton). 

```e```: the energy of the core of the source (parton) in [GeV. It is not used in the current code setting.

```px```: the x component of the momentum of the core of the source (parton) in [GeV]. It is not used in the current code setting.

```py```: the y component of the momentum of the core of the source (parton) in [GeV]. It is not used in the current code setting.

```pz```: the z (NOT eta) component of the momentum of the core of the source (parton) in [GeV]. It is not used in the current code setting.

```de```: the amount of energy loss of the core of the source (parton) in [GeV]. It is NOT time derivative.

```dpx```: the amount of px-loss of the core of the source (parton) in [GeV]. It is NOT time derivative.

```dpy```: the amount of py-loss of the momentum of the core of the source (parton) in [GeV]. It is NOT time derivative.

```dpz```: the amount of pz-loss of the momentum of the core of the source (parton) in [GeV]. It is NOT time derivative.
