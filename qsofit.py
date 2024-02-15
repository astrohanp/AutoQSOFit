#Authors:
#    Rohan Pattnaik
#    Felix Martinez

#### importing packages
# Standard packages
import glob, os, sys, timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

# Scipy packages
import scipy.interpolate as sc
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Fitting packages
from pyqsofit.PyQSOFit import QSOFit
from sklearn.metrics import mean_squared_error as mse
from astropy.convolution import convolve, Gaussian1DKernel

# System packages
from multiprocessing import Pool,cpu_count
import warnings
warnings.filterwarnings("ignore")

# the link to 1d fits in DEIMOS Catalog
fits_link = 'https://irsa.ipac.caltech.edu/data/COSMOS/spectra/deimos/spec1d_fits/'

def cont_sub_alt(df_data):
    '''
    Subtracts the continuum of a spectrum, Attempts to detect emission lines
    of the resulting spectra and returns the best fit model.
    ----
    Input:
    
    df_data: (DataFrame)
        Must have the columns: ['wavelength'] ['flux'] ['err']
        'wavelength' - units of Angstrom
        'flux' - units of of 10^{-17} erg/s/cm^2/Angstrom
        'err' - same units of flux
    ----
    Output:
    
    df_final: (DataFrame)
        Dataframe of the subtracted continuum, has columns: ['wavelength'] ['flux']
        'wavelength' - units of Angstrm
        'flux' - units of of 10^{-17} erg/s/cm^2/Angstrom
        
    best_fit: (array)
        Best Fit model of the resulting spectrum
        
    lines_cent: (dict)
        Emission lines that are trying to be detected, has units of Angstrom
        
    width: (float)
        The range in wavelength in which we are trying to detect emission lines from 'lines_cent'
        
    lines_c_latex: (dict)
        LaTex notation of the 'lines_cent' dictionary
    '''
    
    min_wav = df_data['wavelength'].min()
    max_wav = df_data['wavelength'].max()

    # Pulls models from the PyQSOFit folder, must be adjascent to this file in directory
    models = []
    for i in glob.glob('../pyqsofit/bc03/*.gz'):
        gal_temp = np.genfromtxt(i)
        w = gal_temp[:, 0]
        fl = gal_temp[:, 1]
        df = pd.DataFrame(data=list(zip(w,fl)),columns=['wavelength','flux'])
        df = df[np.logical_and(df['wavelength']>= min_wav,df['wavelength']<= max_wav)]
        models.append(df)

    # Convolve models to velocity dispersion of data and interpolate to match data
    models_conv = []
    bc03_pix = 70.0       # Size of 1 model pixel in km/s
    bc03_vdisp = 75.0     # Approximate velocity of BC03 models
    vdisp = 157.3         # Measured velocity dispersion

    # Create Gaussian kernel
    sigma_pix = np.sqrt(vdisp**2-bc03_vdisp**2)/bc03_pix
    kernel = Gaussian1DKernel(sigma_pix,x_size=np.round(4*sigma_pix+1))

    for each in models:
        conv_flux = convolve(each['flux'].values,kernel,boundary='wrap',normalize_kernel=True)
        df_conv = pd.DataFrame(data=list(zip(each['wavelength'].values,conv_flux)),columns=['wavelength','flux'])
        models_conv.append(df_conv)

    # Specify Line centers
    width = 80.0

    lines_cent = {
        'Lya': 1215.400, 'CIV': 1549.480, 'MgII': 2799.117, 'NeV': 3346.800,
        'NeVI': 3426.850, 'OII': 3729.875, 'NeIII': 3869.000, 'HeIb': 3889.500,
        'Hd': 4102.892, 'Hg': 4341.680, 'OIIIx': 4364.436, 'HeII': 4687.500,
        'Hb': 4862.680, 'OIIIa': 4960.295, 'OIIIb': 5008.240, 'HeI': 5877.200,
        'OI': 6302.046, 'NIIa': 6548.100, 'Ha': 6564.610, 'NIIb': 6583.500,
        'SIIa': 6718.290, 'SIIb': 6732.670, 'SIIIa': 9068.600, 'SIIIb': 9539.600
    }
    lines_c_latex = {
        r'Ly$\alpha$': 1215.400, r'C $\mathrm{IV}$': 1549.480, r'Mg $\mathrm{II}$': 2799.117, r'Ne $\mathrm{V}$': 3346.800,
        r'Ne $\mathrm{VI}$': 3426.850, r'[O $\mathrm{II}$]': 3729.875, r'[Ne $\mathrm{III}$]': 3869.000, r'He $\mathrm{I}$b': 3889.500,
        r'H$\delta$': 4102.892, r'H$\gamma$': 4341.680, r'O $\mathrm{III}$x': 4364.436, r'He $\mathrm{II}$': 4687.500,
        r'H$\beta$': 4862.680, r'[O $\mathrm{III}$]a': 4960.295, r'[O $\mathrm{III}$]b': 5008.240, r'He $\mathrm{I}$': 5877.200,
        r'[O $\mathrm{I}$]': 6302.046, r'[N $\mathrm{II}$]a': 6548.100, r'H$\alpha$': 6564.610, r'[N $\mathrm{II}$]b': 6583.500,
        r'[S $\mathrm{II}$]a': 6718.290, r'[S $\mathrm{II}$]b': 6732.670, r'S $\mathrm{III}$a': 9068.600, r'S $\mathrm{III}$b': 9539.600
    }

    # Mask emission lines in spectra
    mask = []
    for i in range(0,len(df_data)):
        val = 0
        for each in lines_cent.values():
            if df_data['wavelength'].iloc[i] > each-width and df_data['wavelength'].iloc[i] < each+width:
                val = 1
                break
        mask = np.append(mask,val)

    # our resulting masked emission lined arrays
    msk_flux = np.ma.masked_array(df_data['flux'], mask=mask)
    msk_wave = np.ma.masked_array(df_data['wavelength'], mask=mask)

    # Interpolate the missing points of the masked curve
    intp = sc.interp1d(msk_wave.data,msk_flux.data)
    intp_flux = intp(df_data['wavelength'])
    intp_flux = np.nan_to_num(intp_flux)

    best_error = np.inf
    for i,each in enumerate(models_conv):
        
        def intp_func(x,k):
            '''
            Integration function for each of our models, returns the best fit function
            '''
            func = sc.interp1d(each['wavelength'],each['flux']*k,fill_value="extrapolate")
            return func(x)
        
        popt, pcov = curve_fit(intp_func,df_data['wavelength'],intp_flux)           # ,sigma=df_data['error']
        e = mse(intp_flux,intp_func(df_data['wavelength'],popt[0]))
        if e < best_error:
            best_error = e
            best_fit = intp_func(df_data['wavelength'],popt[0])
            best_func = (i,intp_func)
        else:
            pass

    # Performing the subtraction
    flux_sub_cont = df_data['flux'] - best_fit

    # Save the continuum subtracted spectrum in a txt file
    df_final = pd.DataFrame()
    df_final['wavelength'] = df_data['wavelength']
    df_final['flux'] = flux_sub_cont

    return(df_final, best_fit, lines_cent, width, lines_c_latex)

def qfit(row):
    '''
    Fits a galaxy's spectra to emission lines and saves the results in
    an output folder.
    ---
    Input:
    row (DataFrame)
        A catalog of the galaxys you want to fit. row must include these
        columns IN THE FOLLOWING ORDER:
        'Unnamed: 0': (int)
            The index column of the catalog
        'ID': (string)
            The galaxy's ID, must match DEIMOS ID's
        'ra': (float)
            The galaxy's RA
        'dec': (float)
            The galaxy's Dec
        'zspec': (float)
            The galaxy's redshift
        'Qf': (int)
            The quality factor of the galaxy's spectra, this is determined through DEIMOS's catalog
        'fits1d': (string)
            The filename of the 1d spectral fit, must match DEIMOS filename
    ----
    Output:
    Saved results of the model, spectrum, errors, and figure in galaxy folder
    '''

    try:
        # Actual function begins here
        
        ########## Identifying the galaxy in the catalog that was read in ##########
        FARMER_ID = str(row[0])
        DEIMOS_ID = str(row[1])
        ra = row[2]
        dec = row[3]
        z = row[4]
        qf = str(row[5])
        filename = row[6]
        
        
        
        ########## reading in the fits file ##########
        f = fits.open(fits_link+filename) # Close this

        # Loading in the galaxies data
        wl = f[1].data['LAMBDA'][0]
        flux = f[1].data['FLUX'][0]#/1e-17 # Convert to unit of of 10^{-17} erg/s/cm^2/Angstrom for PYQSOFIT input
        err = f[1].data['IVAR'][0]#/1e-17
        df_data = pd.DataFrame(data=list(zip(wl,flux,err)),columns=['wavelength','flux','err'])
        df_data['wavelength'] = df_data['wavelength']/(1+z)
        
        # closing the fits file
        f.close()
        
        
        
        ########## Get results from alternate continuum subtraction ##########
        df_final, best_fit, lines_cent, width, lines_c_latex = cont_sub_alt(df_data)
        
        
        
        ########## Setting up pathways ##########
        path_ex = '.'
        path_out = 'data/results/'
        
        # Creates a results folder to save output
        results_folder = 'results'
        if os.path.exists(results_folder):
            pass
        else:
            os.mkdir(results_folder)
            
        # Creates a galaxy ID folder in your results folder to save output
        fname = FARMER_ID+'_'+DEIMOS_ID+'_'+qf
        folder = results_folder+'/'+fname+'/'
        if os.path.exists(folder):
            pass
        else:
            os.mkdir(folder)


        ########## Prepairing the Data for fitting ##########
        # Setting wavelength to be in rest frame
        wavelength = df_final.wavelength.to_numpy()*(1+z)
        flux = df_final.flux.to_numpy()

        q = QSOFit(wavelength, flux, err, z, ra = ra, dec = dec, path = path_ex)
        
        ########## Do the fitting ##########
        start = timeit.default_timer()
        
#        q.Fit(name=None,  # customize the name of given targets. Default: plate-mjd-fiber
#          # prepocessing parameters
#          nsmooth=1,  # do n-pixel smoothing to the raw input flux and err spectra
#          and_mask=False,  # delete the and masked pixels
#          or_mask=False,  # delete the or masked pixels
#          reject_badpix=False,  # reject 10 most possible outliers by the test of pointDistGESD
#          deredden=True,  # correct the Galactic extinction
#          wave_range=None,  # trim input wavelength
#          wave_mask=None,  # 2-D array, mask the given range(s)
#
#          # host decomposition parameters
#          decompose_host=False,  # If True, the host galaxy-QSO decomposition will be applied
#          host_line_mask=True,
#          # If True, the line region of galaxy will be masked when subtracted from original spectra
#          #BC03=False,  # If True, Bruzual1 & Charlot 2003 host model will be used
#          #Mi=None,  # i-band absolute magnitude, for further decide PCA model, not necessary
#          npca_gal=5,  # The number of galaxy PCA components / galaxy models applied
#          npca_qso=20,  # The number of QSO PCA components applied
#
#          # continuum model fit parameters
#          Fe_uv_op=True,  # If True, fit continuum with UV and optical FeII template
#          poly=False,  # If True, fit continuum with the polynomial component to account for the dust reddening
#          BC=False,  # If True, fit continuum with Balmer continua from 1000 to 3646A
#          initial_guess=None,  # Initial parameters for continuum model, read the annotation of this function for detail
#          rej_abs_conti=False,  # If True, it will iterately reject 3 sigma outlier absorption pixels in the continuum
#          n_pix_min_conti=100,  # Minimum number of negative pixels for host continuuum fit to be rejected.
#
#          # emission line fit parameters
#          linefit=True,  # If True, the emission line will be fitted
#          rej_abs_line=False,
#          # If True, it will iterately reject 3 sigma outlier absorption pixels in the emission lines
#
#          # fitting method selection
#          MC=True,
#          # If True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array
#          MCMC=False,
#          # If True, do Markov Chain Monte Carlo sampling of the posterior probability densities to produce the error array
#          nsamp=200,
#          # The number of trials of the MC process (if MC=True) or number samples to run MCMC chain (if MCMC=True)
#
#          # advanced fitting parameters
#          param_file_name='qsopar.fits',  # Name of the qso fitting parameter FITS file.
#          nburn=20,  # The number of burn-in samples to run MCMC chain
#          nthin=10,  # To set the MCMC chain returns every n samples
#          epsilon_jitter=1e-5,
#          # Initial jitter for every initial guass to avoid local minimum. (Under test, not recommanded to change)
#
#          # customize the results
#          save_result=False,  # If True, all the fitting results will be saved to a fits file
#          save_fits_name=None,  # The output name of the result fits
#          save_fits_path=path_out,  # The output path of the result fits
#          plot_fig=True,  # If True, the fitting results will be plotted
#          save_fig=False,  # If True, the figure will be saved
#          plot_corner=False,  # Whether or not to plot the corner plot results if MCMC=True
#
#          # debugging mode
#          verbose=False,  # turn on (True) or off (False) debugging output
#
#          # sublevel parameters for figure plot and emcee
#          kwargs_plot={
#              'save_fig_path': path_out,  # The output path of the figure
#              'broad_fwhm'   : 1200  # km/s, lower limit that code decide if a line component belongs to broad component
#          },
#          kwargs_conti_emcee={},
#          kwargs_line_emcee={})

########## Test code that has PyQSOFit Defaults ##########

        q.Fit(name=None,  # customize the name of given targets. Default: plate-mjd-fiber
             # prepocessing parameters
             nsmooth=1,  # do n-pixel smoothing to the raw input flux and err spectra
             and_mask=False,  # delete the and masked pixels
             or_mask=False,  # delete the or masked pixels
             reject_badpix=True,  # reject 10 most possible outliers by the test of pointDistGESD
             deredden=True,  # correct the Galactic extinction
             wave_range=None,  # trim input wavelength
             wave_mask=None,  # 2-D array, mask the given range(s)

             # host decomposition parameters
             decompose_host=False,  # If True, the host galaxy-QSO decomposition will be applied
             host_line_mask=True,
             # If True, the line region of galaxy will be masked when subtracted from original spectra
             BC03=False,  # If True, Bruzual1 & Charlot 2003 host model will be used
             Mi=None,  # i-band absolute magnitude, for further decide PCA model, not necessary
             npca_gal=5,  # The number of galaxy PCA components / galaxy models applied
             npca_qso=10,  # The number of QSO PCA components applied

             # continuum model fit parameters
             Fe_uv_op=True,  # If True, fit continuum with UV and optical FeII template
             poly=True,  # If True, fit continuum with the polynomial component to account for the dust reddening
             BC=False,  # If True, fit continuum with Balmer continua from 1000 to 3646A
             initial_guess=None,  # Initial parameters for continuum model, read the annotation of this function for detail
             rej_abs_conti=False,  # If True, it will iterately reject 3 sigma outlier absorption pixels in the continuum
             n_pix_min_conti=100,  # Minimum number of negative pixels for host continuuum fit to be rejected.

             # emission line fit parameters
             linefit=True,  # If True, the emission line will be fitted
             rej_abs_line=False,
             # If True, it will iterately reject 3 sigma outlier absorption pixels in the emission lines

             # fitting method selection
             MC=True,
             # If True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array
             MCMC=False,
             # If True, do Markov Chain Monte Carlo sampling of the posterior probability densities to produce the error array
             nsamp=200,
             # The number of trials of the MC process (if MC=True) or number samples to run MCMC chain (if MCMC=True)

             # advanced fitting parameters
             param_file_name='qsopar.fits',  # Name of the qso fitting parameter FITS file.
             nburn=20,  # The number of burn-in samples to run MCMC chain
             nthin=10,  # To set the MCMC chain returns every n samples
             epsilon_jitter=0.,
             # Initial jitter for every initial guass to avoid local minimum. (Under test, not recommanded to change)

             # customize the results
             save_result=False,  # If True, all the fitting results will be saved to a fits file
             save_fits_name=None,  # The output name of the result fits
             save_fits_path=path_out,  # The output path of the result fits
             plot_fig=False,  # If True, the fitting results will be plotted
             save_fig=True,  # If True, the figure will be saved
             plot_corner=False,  # Whether or not to plot the corner plot results if MCMC=True

             # debugging mode
             verbose=False,  # turn on (True) or off (False) debugging output

             # sublevel parameters for figure plot and emcee
             kwargs_plot={
                 'save_fig_path': '.',  # The output path of the figure
                 'broad_fwhm'   : 1200  # km/s, lower limit that code decide if a line component belongs to broad component
                 },
             kwargs_conti_emcee={},
             kwargs_line_emcee={})
            
        ########## Creating Title for Figure ##########
        # Add plot title - Commented out plt.title part in PyQSOFIT.py
        plt.title(fname+' (z = '+str(np.round(z,3))+')',fontdict={'fontsize':18})
        plt.savefig(folder+fname+'.png')
        plt.close()

        ########## Creating a DataFrame that holds erros and results ##########
        # setting up gauss results
        all_gauss_name = q.gauss_result_name[np.ix_(*[range(0,i,2) for i in q.gauss_result_name.shape])]
        all_gauss_result = q.gauss_result[np.ix_(*[range(0,i,2) for i in q.gauss_result.shape])]
        all_lines = []
        for a1,a2 in zip(all_gauss_name[np.ix_(*[range(0,i,3) for i in all_gauss_name.shape])],np.reshape(all_gauss_result,(all_gauss_result.shape[0]//3,3))):
            all_lines.append((a1[:-6],a2))

        centerwaves = []
        for each in all_lines:
            centerwaves.append(q.linelist[q.linelist['linename'] == each[0][:-2]]['lambda'][0])
        
        # making the dataframe
        collect = []
        for line,c,i in zip(all_lines, centerwaves, range(0,q.gauss_result_all.T.shape[0],3)):
            if 'br' in line[0]:
                typ = 'broad'
            elif 'w' in line[0]:
                typ = 'broad'
            else:
                typ = 'narrow'
            s_c_s = np.array([[sc,cen,sig] for sc,cen,sig in zip(q.gauss_result_all.T[i],q.gauss_result_all.T[i+1],q.gauss_result_all.T[i+2])])
            collect.append((line[0][:-2],line[0][-1],typ,c,line[1],s_c_s))
        
        collect = pd.DataFrame(data=collect,columns=['Linename','gauss_num','type','compcenter','scale_centerwave_sigma','all_scale_centerwave_sigma'])

        final_lines = []
        for name,group in collect.groupby('Linename'):
            if group.shape[0]>1:
                final_lines.append((group['Linename'].values[0],group.shape[0],
                                    group['type'].values[0],group['compcenter'].values[0],
                                    np.concatenate(group['scale_centerwave_sigma'].values,axis=0),
                                    np.concatenate(group['all_scale_centerwave_sigma'].values,axis = 1)))
            else:
                final_lines.append((group['Linename'].values[0],1,group['type'].values[0],
                                    group['compcenter'].values[0],group['scale_centerwave_sigma'].values[0],
                                    group['all_scale_centerwave_sigma'].values[0]))

        final_lines = pd.DataFrame(data=final_lines,columns=['Linename','ngauss','type','compcenter',
                                                             'scale_centerwave_sigma','all_scale_centerwave_sigma'])
        
        all_fwhms, all_sigmas, all_ews, all_peaks, all_areas, all_snrs = [], [], [], [], [], []

        for index,row in  final_lines.iterrows():

            fwhm, sigma, ew, peak, area, snr = q.line_prop(row['compcenter'], row['scale_centerwave_sigma'], row['type'])
            all_fwhm = np.zeros(np.shape(row['all_scale_centerwave_sigma'])[0])
            all_sigma = np.zeros(np.shape(row['all_scale_centerwave_sigma'])[0])
            all_ew = np.zeros(np.shape(row['all_scale_centerwave_sigma'])[0])
            all_peak = np.zeros(np.shape(row['all_scale_centerwave_sigma'])[0])
            all_area = np.zeros(np.shape(row['all_scale_centerwave_sigma'])[0])
            all_snr = np.zeros(np.shape(row['all_scale_centerwave_sigma'])[0])

            for k, s in enumerate(row['all_scale_centerwave_sigma']):
                s1 = np.array([s[0],row['scale_centerwave_sigma'][1],s[2]])
        
                try:
                    all_fwhm[k], all_sigma[k], all_ew[k], all_peak[k], all_area[k], all_snr[k] = q.line_prop(row['compcenter'], s1, row['type'])
                except:
                    pass
                    
            ########## Performing Statistics on our Data ##########
            fwhm_std = np.std(all_fwhm)
            sigma_std = np.std(all_sigma)
            ew_std = np.std(all_ew)
            peak_std = np.std(all_peak)
            area_std = np.std(all_area)
            snr_std = np.std(all_snr)

            # Prints the outputs
#            print("FWHM (km/s):",fwhm,fwhm_std)
#            print("Sigma (km/s):",sigma,sigma_std)
#            print("EW (A):",ew,ew_std)
#            print("Peak (A):",peak,peak_std)
#            print("area (10^(-17) erg/s/cm^2):",area,area_std)
    
            # Saving the FWHM and it's std
            final_lines.loc[index,'FWHM'] = fwhm
            all_fwhms.append(all_fwhm)
            final_lines.loc[index,'FWHM_err'] = fwhm_std

            # Saving the Sigma and it's std
            final_lines.loc[index,'Sigma'] = sigma
            all_sigmas.append(all_sigma)
            final_lines.loc[index,'Sigma_err'] = sigma_std

            # Saving the EW and it's std
            final_lines.loc[index,'EW'] = ew
            all_ews.append(all_ew)
            final_lines.loc[index,'EW_err'] = ew_std

            # Saving the Peak and it's std
            final_lines.loc[index,'Peak'] = peak
            all_peaks.append(all_peak)
            final_lines.loc[index,'Peak_err'] = peak_std

            # Saving the Flux and it's std
            final_lines.loc[index,'flux'] = area
            all_areas.append(all_area)
            final_lines.loc[index,'flux_err'] = area_std

            # Saving the SNR and it's std
#            final_lines.loc[index,'snr'] = snr
#            all_snrs.append(all_snr)
#            final_lines.loc[index,'snr_err'] = snr_std
    
            if area_std>0:
                snr = area/area_std
            else:
                snr = 'nan'
            final_lines.loc[index,'snr'] = snr
    
        final_lines['all_FWHM'] = all_fwhms
        final_lines['all_Sigma'] = all_sigmas
        final_lines['all_EW'] = all_ews
        final_lines['all_Peak'] = all_peaks
        final_lines['all_flux'] = all_areas
        #final_lines['all_snr'] = all_snrs
        final_lines.to_pickle(folder+fname+'_table.pkl')

        ########## Saving the results ##########
        # Save gauss results
        gauss_results = np.array([q.gauss_result,q.gauss_result_name])
        np.save(folder+fname+'_gauss_results.npy',gauss_results)
        np.save(folder+fname+'_many_gauss_linefit.npy',q.Manygauss(np.log(q.wave),q.gauss_result[0::2]))

        # Save other plotting files
        np.save(folder+fname+'_wave_flux_err.npy',np.array([q.wave,q.flux,q.err]))
        np.save(folder+fname+'_f_conti_model.npy',q.f_conti_model)
        np.save(folder+fname+'_wave_flux_org.npy',np.array([wl,flux]))
        
        # ending the timer
        end = timeit.default_timer()
        print(f'Fitting completed for Galaxy',DEIMOS_ID,'saved the results in',folder,f'it took {np.round(end - start, 1)}s')
        
        
        ########## Creating a Catalogue ##########
        # this should overwrite any successive entry for the same source ID.
        output_cat = 'results_cat.pkl'
        
        try:
            resfile = pd.read_pickle(output_cat)
        except FileNotFoundError:
            resfile = pd.DataFrame([], columns=['ind', 'id', 'Qf', 'ra', 'dec', 'z','wavelength', 'flux', 'flux_err',
                                                'model_flux', 'line_obs', 'line_gauss', 'line_type', 'line_compcenter',
                                                'line_scale_centerwave_sigma', 'line_all_scale_centerwave_sigma',
                                                'line_FWHM', 'line_FWHM_err', 'line_Sigma', 'line_Sigma_err',
                                                'line_EW', 'line_EW_err', 'line_Peak', 'line_Peak_err',
                                                'line_flux', 'line_flux_err', 'line_snr', 'line_all_FWHM',
                                                'line_all_Sigma', 'line_all_EW', 'line_all_Peak', 'line_all_flux'])

        if DEIMOS_ID in resfile.id.values:
            resfile = resfile.loc[~resfile.ind == int(FARMER_ID)]
        resfile.loc[len(resfile.index)] = [int(FARMER_ID), DEIMOS_ID, qf, ra, dec, z, q.wave, q.flux, q.err,
                                          q.f_conti_model,
                                          final_lines.Linename.to_numpy(),
                                          final_lines.ngauss.to_numpy(),
                                          final_lines.type.to_numpy(),
                                          final_lines.compcenter.to_numpy(),
                                          final_lines.scale_centerwave_sigma.to_numpy(),
                                          final_lines.all_scale_centerwave_sigma.to_numpy(),
                                          final_lines.FWHM.to_numpy(),
                                          final_lines.FWHM_err.to_numpy(),
                                          final_lines.Sigma.to_numpy(),
                                          final_lines.Sigma_err.to_numpy(),
                                          final_lines.EW.to_numpy(),
                                          final_lines.EW_err.to_numpy(),
                                          final_lines.Peak,
                                          final_lines.Peak_err.to_numpy(),
                                          final_lines.flux.to_numpy(),
                                          final_lines.flux_err.to_numpy(),
                                          final_lines.snr.to_numpy(),
                                          final_lines.all_FWHM.to_numpy(),
                                          final_lines.all_Sigma.to_numpy(),
                                          final_lines.all_EW.to_numpy(),
                                          final_lines.all_Peak.to_numpy(),
                                          final_lines.all_flux.to_numpy()]

        resfile.to_pickle(output_cat)

        print('Compiled',DEIMOS_ID,'into',output_cat,'catalogue')

    except Exception as e:
        ########## Printing out the error ##########
        FARMER_ID = str(row[0])
        DEIMOS_ID = str(row[1])
        print(FARMER_ID)
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

        #e_file = open('results/'+FARMER_ID+'_'+DEIMOS_ID+'.txt',"w")
        #e_file.write(str('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))+' '+str(type(e).__name__)+' '+str(e))
        #e_file.close()





if __name__ == '__main__':
    # First argument is catalog
    catalog = str(sys.argv[1])
    
    # Starting the timer
    start = timeit.default_timer()
    
    # Loading in the catalog
    cat = pd.read_csv(catalog)
    cat = list(cat[['Unnamed: 0','ID', 'ra', 'dec', 'zspec', 'Qf', 'fits1d']].to_numpy())

    # Prepping for parallel processing
    print('CPU Count: ',cpu_count())
    p = Pool(cpu_count()-1)
    
    # Performing the fit
    p.map(qfit,cat)

    # Ending the timer
    end = timeit.default_timer()
    print(f'Fitting finished in : {np.round(end - start)}s')
