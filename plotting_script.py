# Authors: Rohan Pattnaik, Felix Martinez and the PyQSOFit Team

# importing our packages

# basic packages
import glob, os, sys, timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

# plotting packages
from matplotlib.pyplot import cm
import matplotlib as mpl
from matplotlib.gridspec import *
from matplotlib.ticker import *
import random

# Setting up pathing
path_em = 'results/'
path = os.getcwd()

# Get complete linelist
complete_linelist = fits.open('qsopar.fits')[1].data
bad_id = []
non_detect = []

# reading in the table
table = pd.read_pickle('results_cat.pkl')

# iterating through the catalog and plotting the figures
for i in range(len(table)):
    #for index,row in broad_cat[broad_cat['ID_2'].isin(folder_ids)].iterrows():
    ID = table['id'][i]
    z = table['z'][i]
    fold_name = str(table['ind'][i])+'_'+ID+'_'+str(table['Qf'][i])
    print('Plotting',fold_name)
    
    outpath=path_em+fold_name

    gauss_result = np.load(path_em+fold_name+'/'+fold_name+'_gauss_results.npy')
    gauss_linefit = np.load(path_em+fold_name+'/'+fold_name+'_many_gauss_linefit.npy')
    f_conti_model = np.load(path_em+fold_name+'/'+fold_name+'_f_conti_model.npy')
    wave_fl_err = np.load(path_em+fold_name+'/'+fold_name+'_wave_flux_err.npy')
    wave_fl_org = np.load(path_em+fold_name+'/'+fold_name+'_wave_flux_org.npy')
    res_table = pd.read_pickle(path_em+fold_name+'/'+fold_name+'_table.pkl')
    
    # Get lines that were detected
    scale_idxs = list(np.isin([s for s in gauss_result[1] if "scale" in s],[s for s in gauss_result[1] if "scale_err" in s],invert=True))
    scale_arr = np.array([s for s in gauss_result[1] if "scale" in s])[scale_idxs]
    scale_idxs = np.isin(gauss_result[1],scale_arr)
    non_zero_idx = list(gauss_result[0][scale_idxs].astype(np.float64)>0)

    det_lines_names = [each[:-8] for each in gauss_result[1][scale_idxs][non_zero_idx]]


    # Get detected linelist subset
    det_idxs = list(np.isin(complete_linelist['linename'],det_lines_names))
    det_lines = complete_linelist[det_idxs][complete_linelist[det_idxs]['lambda'].argsort()]
    if len(det_lines) == 0:
        non_detect.append(ID)
        
    df_complist = pd.read_csv('linelist.txt',sep='|')

    comp_det = []
    for each in det_lines_names:
        comp_det.append(df_complist[df_complist['Line Name'] == each])
    comp_det = pd.concat(comp_det,ignore_index=True)
    comp_det = comp_det.sort_values('Wavelength')
        
    det_cent = [each[:-5]+'centerwave' for each in gauss_result[1][scale_idxs][non_zero_idx]]
    det_cent_idx = np.isin(gauss_result[1],det_cent)
    cents = np.exp(np.asarray(gauss_result[0][det_cent_idx],dtype=float))
    cents.sort()
        
    color=list(iter(plt.cm.rainbow(np.linspace(0,1,len(cents)))))
    random.seed(4)
    random.shuffle(color)
    
    # Assign each detected line a color
    l_c = []
    for lam,clr in zip(cents,color):
        l_c.append((lam,clr))
            
    comp_det['new_wl'] = cents

    min_max_comp_subplots = []
    for name,group in comp_det.groupby('Complex name'):
        min_max_comp_subplots.append([np.round(group['new_wl'].min()-30),np.round(group['new_wl'].max()+30)])
    min_max_comp_subplots.sort()
    if len(min_max_comp_subplots)>6:
        min_max_comp_subplots = min_max_comp_subplots[1:]

    # ----- Set up plotting parameters -----
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    
    plt.figure(figsize=[8,3])

    # Outer grid - Galaxy stamp + Entire Spectra
    outer = GridSpec(2,1, height_ratios = [1, 4],hspace=0.01)

    # Calculate the number of subplots to be made
    sub_n = len(min_max_comp_subplots)

    # Top plot
    plt.title(fold_name)

    plt.plot(wave_fl_err[0],wave_fl_err[1],linewidth=0.8,label='Data',zorder=9,c='k')
    plt.plot(wave_fl_err[0],gauss_linefit+f_conti_model,linestyle='dashdot',linewidth=0.8,label='Fit',zorder=10,c='red')
    plt.legend(loc=(0.85, 1.001),fontsize=10)

    xmin = min(wave_fl_err[0])
    xmax = max(wave_fl_err[0])

    plt.xlim(xmin,xmax)

    plt.xticks(np.linspace(xmin,xmax,num=5,endpoint=True))
    minor_locator = AutoMinorLocator(10)

    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    
    plt.xlabel(r'Rest-frame Wavelength ($\AA$)',size=15)
    plt.ylabel(r'Flux ($10^{-17}$ $erg$ $\AA^{-1} s^{-1} cm^{-2}$)', rotation=90, size=15)

    for each in l_c:
        plt.axvline(each[0],color=each[1], linestyle='--',linewidth=0.8)

    plt.subplots_adjust(top=0.94,bottom=0.11,left=0.09,right=0.96,hspace=0.25)
    plt.savefig(outpath+'/'+fold_name+'_simple.png', bbox_inches='tight')
    plt.close()
    print('Complete! Saved',fold_name,'as '+outpath+'/'+fold_name+'_simple.png')
