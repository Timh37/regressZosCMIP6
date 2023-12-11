'''
Created by: Tim Hermans, 11-12-23
'''
import xarray as xr
import numpy as np
import os
import fnmatch

def amean(ds):
    ds = ds.resample(time='1Y').mean(dim='time')
    ds['time'] = ds['time'].dt.year
    return ds

def regress_dzoszostoga_on_dzostoga(dzos_zostoga):
    
    tdim = dzos_zostoga.zostoga.dims[0] #get time dimension
    
    regr_ds = dzos_zostoga.copy(deep=True) #deepcopy dataset
    regr_ds[tdim] = dzos_zostoga.zostoga.values.flatten() #set zostoga as time dimension, for the regression on zostoga
    
    return regr_ds.zoszostoga.polyfit(dim=tdim,deg=1,skipna=True)

#set input directories (to dedrifted zos/zostoga)
zos_dir = '/Volumes/Naamloos/PhD_Data/CMIP6/dedrifted_linear/zos_0mean_1x1/'
zostoga_dir = '/Volumes/Naamloos/PhD_Data/CMIP6/dedrifted_linear/zostoga/'

ref_period = np.arange(1995,2015) #set reference period relative to compute annual anomalies to

out_dir = '/Volumes/Naamloos/PhD_Data/CMIP6/regression/zoszostoga_zostoga' #set output directory

#### find list of models providing both zos/zostoga
models = list(set(os.listdir(zostoga_dir)) & set(os.listdir(zos_dir))) 
models = [m for m in models if '.' not in m] #exclude MRI?
models.remove('MRI-ESM2-0') #some issues with the starting values of this data, see email with VMS
models.sort()

for m,model in enumerate(models): #for each model
    print('Processing: ' + model)
    ssp_files = fnmatch.filter(os.listdir(os.path.join(zostoga_dir,model)),'*ssp*nc') #find available SSP simulations for zostoga (could be multiple variants per ssp)
    unique_ssps = list(set([k.split('_')[3] for k in ssp_files])) #get unique SSPs
    unique_ssps.sort()
    
    model_data = [] #list to append SSP output to for current model
    fns = []
    for s,ssp in enumerate(unique_ssps):#enumerate(unique_ssps): #for each unique SSP
        files = fnmatch.filter(ssp_files,'*'+ssp+'*') #get simulations for that SSP        
        ssp_passed = 0 #track if simulations succesfully loaded in for current SSP
        
        for f,zostoga_ssp_file in enumerate(files): #for each variant of current SSP
            print('Opening: ' + zostoga_ssp_file)
            variant = zostoga_ssp_file.split('_')[4]
        
            try: 
                zos_ssp_file = fnmatch.filter(os.listdir(os.path.join(zos_dir,model)),'*'+ssp+'*'+variant+'*nc')[0]
                zos_hist_file = fnmatch.filter(os.listdir(os.path.join(zos_dir,model)),'*'+'historical'+'*'+variant+'*nc')[0]
                zostoga_hist_file = fnmatch.filter(os.listdir(os.path.join(zostoga_dir,model)),'*'+'historical'+'*'+variant+'*nc')[0]
                
                try: #if simulations run to 2300, historical and ssp are not decoded in the same way so that open_mfdataset fails, so try opening them separately
                    zostoga = xr.open_mfdataset([os.path.join(zostoga_dir,model,zostoga_hist_file),os.path.join(zostoga_dir,model,zostoga_ssp_file)])
                    zos = xr.open_mfdataset([os.path.join(zos_dir,model,zos_hist_file),os.path.join(zos_dir,model,zos_ssp_file)])
                except:
                    zostoga = xr.concat((xr.open_dataset(os.path.join(zostoga_dir,model,zostoga_hist_file),use_cftime=True),
                                         xr.open_dataset(os.path.join(zostoga_dir,model,zostoga_ssp_file),use_cftime=True)),dim='time')
                    zos = xr.concat((xr.open_dataset(os.path.join(zos_dir,model,zos_hist_file),use_cftime=True),
                                         xr.open_dataset(os.path.join(zos_dir,model,zos_ssp_file),use_cftime=True)),dim='time')
            except:
                print('Could not open all files required for regression, trying next simulation.')
                continue
            
            fns.append(zostoga_ssp_file) #keep track of files used
            
            zos_zostoga = xr.merge((zos,zostoga.squeeze())) #merge zos & zostoga into one dataset & assign coordinates
            zos_zostoga = zos_zostoga.resample(time='1Y').mean(dim='time') #compute annual means
            zos_zostoga['time'] = np.linspace(zos_zostoga.time[0].dt.year.values,zos_zostoga.time[-1].dt.year.values,len(zos_zostoga.time)).astype('int') #replace time objects by years to avoid timestamp difference issues
            
            zos_zostoga = zos_zostoga.assign_coords({'ssp':ssp}) #add SSP coordinates
            zos_zostoga = zos_zostoga.expand_dims('ssp')
            
            model_data.append(zos_zostoga.load()) #append ssp data to list of ssp datasets
            
            ssp_passed = 1 #succesfully opened required files for current model/ssp/variant
            
            if ssp_passed: #only use 1 variant per SSP
                break
    
    if len(model_data) == 0:
        continue

    zos_zostoga= xr.concat(model_data,dim='ssp') #combine ssp datasets into one dataset for current model
    zos_zostoga['zoszostoga'] = zos_zostoga['zos'] + zos_zostoga['zostoga'] #create zos+zostoga variable
    
    #compute annual mean anomalies relative to reference period
    dzos_zostoga = (zos_zostoga - zos_zostoga.sel(time=slice(str(ref_period[0]),str(ref_period[-1]))).mean(dim='time')) #compute anomalies relative to reference period
    dzos_zostoga = dzos_zostoga.sel(time=slice(str(ref_period[0]),str(zos_zostoga.time[-1]))) #select period to use for regression
    
    regr_coefs = xr.concat([regress_dzoszostoga_on_dzostoga(dzos_zostoga.sel(ssp=k).dropna(dim='time',subset=['zostoga'])) for k in dzos_zostoga.ssp.values],dim='ssp') #do the regression for each SSP
    regr_coefs = xr.concat((regr_coefs,regress_dzoszostoga_on_dzostoga(zos_zostoga.stack(f=('time','ssp'),create_index=False).dropna(dim='f',subset=['zostoga']))),dim='ssp') #add the regression for all SSPs merged
   
    regr_coefs['ssp'] = np.hstack([zos_zostoga.ssp.values,'merged']) #add SSP coordinate
    regr_coefs.attrs['ref_period'] = ref_period #add some attributes
    regr_coefs.attrs['files_used'] = fns
    
    
    if os.path.exists(os.path.join(out_dir,model)) == False: #store
        os.mkdir(os.path.join(out_dir,model))
    regr_coefs.to_netcdf(os.path.join(out_dir,model,'linregression_coefs.nc'),mode='w')