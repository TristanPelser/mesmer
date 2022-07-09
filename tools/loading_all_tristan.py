#!/usr/bin/env python
# -*- coding: utf-8 -*-


#################
## Functions to process data
#################            
            
import numpy as np
import glob
#import geopandas # to read the shape files (there'd also be cartopy internal fct but there I'm lacking the knowledge)
#from cartopy.io.shapereader import Reader
#import fiona
import datetime
import copy
import mplotutils as mpu
import xarray as xr
import pandas as pd
import cf_units


def norm_cos_wgt(lats):
    
    return np.cos(np.deg2rad(lats))


def load_data_single_mod(gen,model,Tref_all=True,Tref_start='1870-01-01',Tref_end='1900-01-01',usr_time_res="ann"):
	""" Load the all initial-condition members of a single model in cmip5 or cmip6 for given scenario plus associated historical period.

		Keyword argument:
		- gen: generation (cmip5 = 5 and cmip6 = 6 are implemented)
		- model: model str
		- scenario: scenario str
		- Tanglob_idx: decides if wgt Tanglob is computed (and returned) or not, default is not returned
		- Tref_all: decides if the Tref at each grid point is dervied based on all available runs or not, default is yes       
		- Tref_start: starting point for the reference period with default 1870
		- Tref_end: first year to no longer be included in reference period with default 1900

		Output:
		- y: the land grid points of the anomalies of the variable on grid centered over 0 longitude (like the srexgrid) 
		- time: the time slots
		- srex: the gridded srex regions
		- df_srex: data frame containing the shape files of the srex regions
		- lon_pc: longitudes for pcolormesh (needs 1 more than on grid)
		- lat_pc: latitudes for pcolormesh (needs 1 more than on grid)
		- idx_l: array with 0 where sea, 1 where land (assumption: land if frac land > 0)
		- wgt_l: land grid point weights to derive area weighted mean vars
		- Tan_wgt_globmean = area weighted global mean temperature

	"""
    # the dictionaries are NOT ordered properly + some other adjustments -> will need to be careful with my old scripts

    # see e-mail from Verena on 20191112 for additional infos how could read in several files at once with xarr
    # additionally: she transforms dataset into dataarray to make indexing easier -> for consistency reason with earlier
        # version of emulator (& thus to be able to reuse my scripts), I do not do this (fow now).
    
	# right now I keep reloading constants fields for each run I add -> does not really make sense. 
    # Maybe add boolean to decide instead. however they are small & I have to read them in at some point anyways
    # -> maybe path of least resistence is to not care about it
	print('start with model',model)

	# vars which used to be part of the inputs but did not really make sense as I employ the same ones all the time anyways (could be changed later if needed)
	var='tas'
	temp_res = usr_time_res # if not, reading the var file needs to be changed as time var is not named in the same way anymore
	spatial_res = 'g025'


    # load in the constants files
	dir_data = '/home/tristan/mesmer/data/'
	file_ls = 'interim_invariant_lsmask_regrid.nc' # ERA-interim mask regridded by Richard from 73x144 to 72x144
	file_srex = 'srex-region-masks_20120709.srex_mask_SREX_masks_all.25deg.time-invariant.nc'
	file_srex_shape = 'referenceRegions.shp'


	srex_names = ['ALA','CGI','WNA','CNA','ENA','CAM','AMZ','NEB','WSA','SSA','NEU','CEU','MED','SAH','WAF','EAF','SAF',
             'NAS','WAS','CAS','TIB','EAS','SAS','SEA','NAU','SAU'] # SREX names ordered according to SREX mask I am 
                    # employing

	srex_raw = xr.open_mfdataset(dir_data+file_srex, combine='by_coords',decode_times=False) # srex_raw nrs from 1-26
	lons, lats = np.meshgrid(srex_raw.lon.values,srex_raw.lat.values) # the lon, lat grid (just to derive weights)    
    
	frac_l = xr.open_mfdataset(dir_data+file_ls, combine='by_coords',decode_times=False) #land-sea mask of ERA-interim bilinearily interpolated 
	frac_l_raw = np.squeeze(copy.deepcopy(frac_l.lsm.values))
	#frac_1["time"]=pd.to_datetime(frac_1.time.values)
	frac_l = frac_l.where(frac_l.lat>-60,0) # remove Antarctica from frac_l field (ie set frac l to 0)

	idx_l=np.squeeze(frac_l.lsm.values)>0.0 # idex land #-> everything >0 I consider land
 

	wgt = norm_cos_wgt(lats) # area weights of each grid point
	wgt_l = (wgt*frac_l_raw)[idx_l] # area weights for land grid points (including taking fraction land into consideration)
	lon_pc, lat_pc = mpu.infer_interval_breaks(frac_l.lon, frac_l.lat) # the lon / lat for the plotting with pcolormesh
	srex=(np.squeeze(srex_raw.srex_mask.values)-1)[idx_l] # srex indices on land

    
	y={}
	T_ref = np.zeros(idx_l.shape)
	run_nrs={}
	if gen !== 6:
        print("This is only for next gen CMIP6 models!")
            
                  
	if gen == 6:
		dir_var = '/home/tristan/mesmer/CMIP6/tas/%s/g025/'%usr_time_res#'/net/cfc/cmip6/Next_Generation/tas/' #<- switch once stable
		run_names_list=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_ssp*_'+'r*i1p1f*'+'_'+spatial_res+'.nc'))
		run_names_list_historical=sorted(glob.glob(dir_var+var+'_'+temp_res+'_'+model+'_historical_'+'r*i1p1f*'+'_'+spatial_res+'.nc'))

		if model=='CESM2-WACCM':
			run_names_list.remove('/home/tristan/mesmer/CMIP6/tas/%s/g025/tas_%s_CESM2-WACCM_ssp585_r4i1p1f1_g025.nc'%(usr_time_res,usr_time_res))
			run_names_list.remove('/home/tristan/mesmer/CMIP6/tas/%s/g025/tas_%s_CESM2-WACCM_ssp585_r5i1p1f1_g025.nc'%(usr_time_res,usr_time_res))
		for run_name in run_names_list:
			run_name_ssp = run_name
			data = xr.open_mfdataset(run_name_ssp,concat_dim='time').sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180))  # assign_coords so same labels as others
			scen = run_name.split('/')[-1].split('_')[-3]
			run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0])                   
			if scen not in list(y.keys()):
				y[scen]={}
				run_nrs[scen]=[]
				y[scen][run] = data.tas.values # still absolute values + still contains sea pixels
				run_nrs[scen].append(run) 
			else:                         
				y[scen][run] = data.tas.values # still absolute values + still contains sea pixels
				run_nrs[scen].append(run) 
		y['historical']={}   
		run_nrs['historical']=[]  
		for run_name in run_names_list_historical:
			run_name_hist = run_name
			data = xr.open_mfdataset(run_name_hist,concat_dim='time').sel(time=slice('1870-01-01', '2101-01-01')).roll(lon=72)
			data = data.assign_coords(lon= (((data.lon + 180) % 360) - 180))  # assign_coords so same labels as others
			run = int(run_name.split('/')[-1].split('_')[-2].split('r')[1].split('i')[0])                          
			y['historical'][run] = data.tas.values # still absolute values + still contains sea pixels
			run_nrs['historical'].append(run) 
            
            
			T_ref += data.tas.sel(time=slice(Tref_start, Tref_end)).mean(dim='time').values*1.0/len(run_names_list_historical) # sum up all ref climates 
			y['historical'][run]=y['historical'][run][:,idx_l]
       # obtain the anomalies
	for scen in [i for i in y.keys()]:
		print(scen, run_nrs[scen])     
		for run in run_nrs[scen]:
			if Tref_all == True:
				try:
					y[scen][run]=(y[scen][run]-T_ref)[:,idx_l]
				except: 
					y[scen][run]=(y[scen][run]-T_ref[idx_l]) 
					print('exception dealt with, ', scen,y[scen][run].shape)
			else:
				#print(y[scen][run].shape)
				try:
					y[scen][run]=y[scen][run][:,idx_l]#-T_ref_1)[:,idx_l]  
				except:
					y[scen][run]=y[scen][run]                    
                    
	if (data.lon!=srex_raw.lon).any() and (srex_raw.lon!=frac_l.lon).any():
		print('There is an error. The grids do not agree.')
	time=data["time"]
	if Tref_all == False:
		return y, run_nrs #df_srex,
	else:
		return y,T_ref, run_nrs#df_srex,            
        
