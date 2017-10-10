# load same modules as the 'strongconstraint_utils.py
import json
import os
import platform
import subprocess
import glob
import copy

import matplotlib.pyplot as plt
import numpy as np

import gp_emulator
import prosail
from SCOPE_tools import SCOPE

import time                             # required for StandardStateCMEM and grab_cmem_emulators
from collections import OrderedDict     # required for StandardStateCMEM
from eoldas_ng import State, MetaState  # required for StandardStateCMEM
from eoldas_ng import ObservationOperatorTimeSeriesGP

os.chdir('/home/ucfajti/Simulations/Python/scope/')
import math

def grab_cmem_emulators ( vza, sza, raa, emulator_home="./emulator3/", vza_q=10, sza_q=10, raa_q=30,ntrain=200,thresh=0.97):
    # Locate all available emulators...
    files = glob.glob("%s*.npz" % emulator_home)
    emulator_search_dict = {}
    for f in files:
        try:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[2]),
                                float(f.split("/")[-1].split("_")[1]) - \
                                float(f.split("/")[-1].split("_")[3]) ] = f
        except:
            emulator_search_dict[  float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[1]),
                                float(f.split("/")[-1].split("_")[2]) ] = f

    # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
    # Remove some weirdos...
    emu_keys = np.array( emulator_search_dict.keys() )
    emulators = {}
    for i, isza in enumerate ( sza ):
        isza  = int(sza_q * round(float(sza[i])/sza_q))
        ivza  = int(vza_q * round(float(vza[i])/vza_q))
        iraa  = int(raa_q * round(float(raa[i])/raa_q))

        thiskey = (isza, ivza, iraa)
        #try:
        #gp = gp_emulator.MultivariateEmulator (
        #dump=emulator_search_dict[ thiskey ] )


        if emu_keys.shape[0] == 0:
            gp = create_cmem_emulators ( isza, ivza, iraa,ntrain=ntrain,thresh=thresh )
            gp.dump_emulator ( os.path.join ( emulator_home,
                                             "%04d_%04d_%04d_cmem" % ( isza, ivza, iraa )))
        elif thiskey in  emulator_search_dict.keys():
            gp = gp_emulator.MultivariateEmulator (
                dump=emulator_search_dict[ thiskey ] )
        else:
            gp = create_cmem_emulators ( isza, ivza, iraa ,ntrain=ntrain,thresh=thresh)
            gp.dump_emulator ( os.path.join ( emulator_home,
                                     "%04d_%04d_%04d_cmem" % ( isza, ivza, iraa )))

        emulators[isza, ivza, iraa ] = gp
    return emulators

def grab_cmem_emulators2 ( vza, sza, raa, emulator_home="./emulator3/", vza_q=10, sza_q=10, raa_q=30,ntrain=200,thresh=0.97):
    # Locate all available emulators...
    files = glob.glob("%s*.npz" % emulator_home)
    emulator_search_dict = {}
    for f in files:
        try:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[2]),
                                float(f.split("/")[-1].split("_")[1]) - \
                                float(f.split("/")[-1].split("_")[3]) ] = f
        except:
            emulator_search_dict[  float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[1]),
                                float(f.split("/")[-1].split("_")[2]) ] = f

    # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
    # Remove some weirdos...
    emu_keys = np.array( emulator_search_dict.keys() )
    emulators = {}
    for i, isza in enumerate ( sza ):
        isza  = int(sza_q * round(float(sza[i])/sza_q))
        ivza  = int(vza_q * round(float(vza[i])/vza_q))
        iraa  = int(raa_q * round(float(raa[i])/raa_q))

        thiskey = (isza, ivza, iraa)
        #try:
        #gp = gp_emulator.MultivariateEmulator (
        #dump=emulator_search_dict[ thiskey ] )


        if emu_keys.shape[0] == 0:
            gp = create_cmem_emulators2 ( isza, ivza, iraa,ntrain=ntrain,thresh=thresh )
            gp.dump_emulator ( os.path.join ( emulator_home,
                                             "%04d_%04d_%04d_cmem" % ( isza, ivza, iraa )))
        elif thiskey in  emulator_search_dict.keys():
            gp = gp_emulator.MultivariateEmulator (
                dump=emulator_search_dict[ thiskey ] )
        else:
            gp = create_cmem_emulators2 ( isza, ivza, iraa ,ntrain=ntrain,thresh=thresh)
            gp.dump_emulator ( os.path.join ( emulator_home,
                                     "%04d_%04d_%04d_cmem" % ( isza, ivza, iraa )))

        emulators[isza, ivza, iraa ] = gp
    return emulators

def grab_prosail_emulators ( vza, sza, raa, emulator_home="./emulator3_sail/", vza_q=10, sza_q=10, raa_q=30,ntrain=200,thresh=0.97):
    # Locate all available emulators...
    files = glob.glob("%s*.npz" % emulator_home)
    emulator_search_dict = {}
    for f in files:
        try:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[2]),
                                float(f.split("/")[-1].split("_")[1]) - \
                                float(f.split("/")[-1].split("_")[3]) ] = f
        except:
            emulator_search_dict[  float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[1]),
                                float(f.split("/")[-1].split("_")[2]) ] = f
    # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
    # Remove some weirdos...
    emu_keys = np.array( emulator_search_dict.keys() )
    emulators = {}
    for i, isza in enumerate ( sza ):
        isza  = int(sza_q * round(float(sza[i])/sza_q))
        ivza  = int(vza_q * round(float(vza[i])/vza_q))
        iraa  = int(raa_q * round(float(raa[i])/raa_q))

        thiskey = (isza, ivza, iraa)
        #try:
        #gp = gp_emulator.MultivariateEmulator (
        #dump=emulator_search_dict[ thiskey ] )


        if emu_keys.shape[0] == 0:
            gp = create_prosail_emulators ( isza, ivza, iraa,ntrain=ntrain,thresh=thresh )
            gp.dump_emulator ( os.path.join ( emulator_home,
                                             "%04d_%04d_%04d_prosail" % ( isza, ivza, iraa )))
        elif thiskey in  emulator_search_dict.keys():
            gp = gp_emulator.MultivariateEmulator (
                dump=emulator_search_dict[ thiskey ] )
        else:
            gp = create_prosail_emulators ( isza, ivza, iraa,ntrain=ntrain,thresh=thresh )
            gp.dump_emulator ( os.path.join ( emulator_home,
                                     "%04d_%04d_%04d_prosail" % ( isza, ivza, iraa )))

        emulators[isza, ivza, iraa ] = gp
    return emulators

def grab_scope_emulators ( vza, sza, raa, emulator_home="./emulator3_scope/", vza_q=10, sza_q=10, raa_q=30,ntrain=400,thresh=0.9999):
    # Locate all available emulators...
    files = glob.glob("%s*_scope*.npz" % emulator_home)
    emulator_search_dict = {}
    for f in files:
        try:
            emulator_search_dict[ float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[2]),
                                float(f.split("/")[-1].split("_")[1]) - \
                                float(f.split("/")[-1].split("_")[3]) ] = f
        except:
            emulator_search_dict[  float(f.split("/")[-1].split("_")[0]), \
                                float(f.split("/")[-1].split("_")[1]),
                                float(f.split("/")[-1].split("_")[2]) ] = f
    # So we have a dictionary inddexed by SZA, VZA and RAA and mapping to a filename
    # Remove some weirdos...
    emu_keys = np.array( emulator_search_dict.keys() )
    emulators = {}
    for i, isza in enumerate ( sza ):
        isza  = int(sza_q * round(float(sza[i])/sza_q))
        ivza  = int(vza_q * round(float(vza[i])/vza_q))
        iraa  = int(raa_q * round(float(raa[i])/raa_q))

        thiskey = (isza, ivza, iraa)
        #try:
        #gp = gp_emulator.MultivariateEmulator (
        #dump=emulator_search_dict[ thiskey ] )


        if emu_keys.shape[0] == 0:
            gp = create_scope_emulators ( isza, ivza, iraa,ntrain=ntrain,thresh=thresh )
            gp.dump_emulator ( os.path.join ( emulator_home,
                                             "%04d_%04d_%04d_scope" % ( isza, ivza, iraa )))
        elif thiskey in  emulator_search_dict.keys():
            gp = gp_emulator.MultivariateEmulator (
                dump=emulator_search_dict[ thiskey ] )
        else:
            gp = create_scope_emulators ( isza, ivza, iraa,ntrain=ntrain,thresh=thresh )
            gp.dump_emulator ( os.path.join ( emulator_home,
                                     "%04d_%04d_%04d_scope" % ( isza, ivza, iraa )))

        emulators[isza, ivza, iraa ] = gp
    return emulators


def create_cmem_emulators ( sza, vza, raa,ntrain=200,thresh=0.97):
    # For cmem the approximate number of baseline functions (using all the variables) are given below
    # Note that for creating for 400 trainingsets already is time consuming while this may not prove advantage at
    # all: Although using more trainingsets increases the variabilty in the trainingsets, they can cause the emulator
    # to be too complex for actual usage and even lead to unstable emulators.

    # Also note that when using emulating the anomalies instead of the actual values, more basis functions are
    # required (which lead to higher computation times). This is as the variance that the emulators then have to
    # achieve is dominated by anomalies and not the overall curve of the spectrum.

    # #######################################
    # Ntrain = 400, thress=0.99, rho_train -                => ~10 basis functions
    # Ntrain = 400, thress=0.99, rho_train - rho_train_m    => ~20 basis functions
    # Ntrain = 200, thress=0.99, rho_train -                => ~10 basis functions
    # Ntrain = 200, thress=0.99, rho_train - rho_train_m    => ~20 basis functions
    # Ntrain = 200, thress=0.97, rho_train -                => ~05 basis functions
    # Ntrain = 200, thress=0.97, rho_train - rho_train_m    => ~10 basis functions
    # #######################################

    # 0.0 read the minimum and maximum values of the specified parameters that are going to be retrieved
    # parameters = ['LAI2',     'SL2',     'CLMP2',   'SZ2',     'ELN2',
    #               'THM2',     'NRATIO2', 'SLW2',    'CW2',     'CAB2',    'CDM2',    'CSEN2',   'XN2',
    #               'LAI1',     'SL1',     'CLMP1',   'ELN1',
    #               'THM1',     'NRATIO1', 'SLW1',    'CW1',     'CAB1',    'CSEN1',   'CDM1',    'CXC1',     'XN1',
    #               'S1',       'S2']

    parameters = ['fwc_lsm_i1','ftl_lsm_i1','fs_laiL_i']

    min_vals =  [0.00000, 150.000, 0.]
    max_vals =  [1.00000, 350.000, 10.]

    # 1.0 create training and validation parameter set
    training_set, distributions = gp_emulator.create_training_set ( parameters, min_vals, max_vals, n_train=ntrain )
    validate_set = gp_emulator.create_validation_set( distributions )

    # 2.0 create set for training Emulators
    Obs_train = []
    for xx in training_set:
        output = do_cmem_fwd_model (xx, vza)
        hpol= float(output[0][0])
        vpol = float(output[1][0])
        Obs_train.append([hpol,vpol])
    Obs_train = np.array ( Obs_train )#[:,:,1]
    wl        = [1,2] #0.2*1e6

    # calculation of mean value of the training_set
    Obs_train_m = Obs_train.mean(axis=0)

    # storing the mean in the stored emulator has not been accomplished in this version. As such this
    # will provide problems when implementing these emulators in eoldas. At the moment therefore the
    # mean value is not used (set to zero)
    Obs_train_m = Obs_train_m*0


    # #######################################

    # 3.0 emulate the forward model using the training set
    gp = gp_emulator.MultivariateEmulator( X=Obs_train - Obs_train_m, y=training_set, thresh=thresh)

    # 4.0 Create validation dataset
    # Obs_validate = []
    # for xx in validate_set:
    #     output = do_cmem_fwd_model (xx, vza)
    #     hpol= float(output[0][0])
    #     vpol = float(output[1][0])
    #     Obs_validate.append([hpol,vpol])
    # Obs_validate = np.array ( Obs_validate )

    # 5.0 Predict validation dataset with emulator
    # Obs_emulated = []
    # for ii in xrange ( validate_set.shape[0]):
    #     Obs_emulated.append ( gp.predict(validate_set[ii])[0] + Obs_train_m)
    # Obs_emulated = np.array (Obs_emulated)

    # 6.0 Validate emulator by cross comparison statistics
    # Error
    # err = Obs_emulated - Obs_validate
    #
    # # RMSE
    # #AE = err.__abs__()
    # SE = np.square(err)
    # MSE = np.mean(SE,0)
    # RMSE = np.sqrt(MSE)
    #
    # # mean error
    # ME = np.mean(err,0)
    #
    # #Maximum Absolute Error
    # #MAE = np.max(AE)
    #
    # # Quantiles Errors
    # qua = np.percentile ( err, [5,25,50,75,95], axis=0 )
    # AE = err.__abs__()
    # MSA = np.max(AE)


    # 7.0 Plot differences
    #import matplotlib.pyplot as plt
    # string = 'Ntrain=%02d, Thresh=%04.3f, MSA=%6.4f.png' %(ntrain, thresh, MSA)
    # filename1 = 'CreateEmulator_Quantiles_(Ntrain_%02d)_(Thresh_%04.3f).png' %(ntrain, thresh)
    # filename2 = 'CreateEmulator_RMSE_(Ntrain_%02d)_(Thresh_%04.3f).png' %(ntrain, thresh)
    #
    # plt.figure()
    # plt.fill_between ( wl, qua[0] ,qua[4], color="0.8")
    # plt.fill_between ( wl, qua[1] ,qua[3], color="0.6")
    # plt.plot ( wl, err.mean(axis=0), '-r')
    # plt.plot ( wl, qua[2], '--g')
    # plt.axhline(0, c="y")
    # plt.axhline(-0.01, c="y")
    # plt.axhline(+0.01, c="y")
    # plt.ylim(-0.05,0.05)
    # #plt.plot(wv, mu_x )
    # plt.xlim(400,2500)
    # plt.title(string)
    #
    # plt.savefig(filename1)
    # plt.close()
    #
    # plt.figure()
    # plt.semilogy ( wl, RMSE, '-r')
    # plt.semilogy ( wl, ME, '-b')
    # plt.semilogy ( wl, -ME, '--b')
    # plt.axhline(+0.01, c="y")
    # plt.ylabel('RMSE [-]')
    # plt.xlim(400,2500)
    # plt.axhline(+0.01, c="y")
    # plt.ylim(1e-5,1e-0)
    # plt.legend(['RMSE', 'Mean Error', '-Mean Error','Upper Thresh'])
    #
    # plt.savefig(filename2)
    # plt.close()

    # import pdb
    # pdb.set_trace()
    return gp

def create_cmem_emulators2 ( sza, vza, raa,ntrain=200,thresh=0.97):
    # For cmem the approximate number of baseline functions (using all the variables) are given below
    # Note that for creating for 400 trainingsets already is time consuming while this may not prove advantage at
    # all: Although using more trainingsets increases the variabilty in the trainingsets, they can cause the emulator
    # to be too complex for actual usage and even lead to unstable emulators.

    # Also note that when using emulating the anomalies instead of the actual values, more basis functions are
    # required (which lead to higher computation times). This is as the variance that the emulators then have to
    # achieve is dominated by anomalies and not the overall curve of the spectrum.

    # #######################################
    # Ntrain = 400, thress=0.99, rho_train -                => ~10 basis functions
    # Ntrain = 400, thress=0.99, rho_train - rho_train_m    => ~20 basis functions
    # Ntrain = 200, thress=0.99, rho_train -                => ~10 basis functions
    # Ntrain = 200, thress=0.99, rho_train - rho_train_m    => ~20 basis functions
    # Ntrain = 200, thress=0.97, rho_train -                => ~05 basis functions
    # Ntrain = 200, thress=0.97, rho_train - rho_train_m    => ~10 basis functions
    # #######################################

    # 0.0 read the minimum and maximum values of the specified parameters that are going to be retrieved
    # parameters = ['LAI2',     'SL2',     'CLMP2',   'SZ2',     'ELN2',
    #               'THM2',     'NRATIO2', 'SLW2',    'CW2',     'CAB2',    'CDM2',    'CSEN2',   'XN2',
    #               'LAI1',     'SL1',     'CLMP1',   'ELN1',
    #               'THM1',     'NRATIO1', 'SLW1',    'CW1',     'CAB1',    'CSEN1',   'CDM1',    'CXC1',     'XN1',
    #               'S1',       'S2']

    parameters = ['fwc_lsm_i1','ftl_lsm_i1','fs_laiL_i','theta']

    min_vals =  [0.00000, 150.000,  0, 0]
    max_vals =  [1.00000, 350.000, 10, 85]

    # 1.0 create training and validation parameter set
    training_set, distributions = gp_emulator.create_training_set ( parameters, min_vals, max_vals, n_train=ntrain )
    validate_set = gp_emulator.create_validation_set( distributions )

    # 2.0 create set for training Emulators
    Obs_train = []
    for xx in training_set:

        theta=xx[3]
        output = do_cmem_fwd_model (xx[0:3], theta)
        print theta
        print output

        hpol= float(output[0][0])
        vpol = float(output[1][0])
        Obs_train.append([hpol,vpol])
    Obs_train = np.array ( Obs_train )#[:,:,1]
    wl        = [1,2] #0.2*1e6

    Obs_train_m = Obs_train.mean(axis=0)


    # #######################################

    # 3.0 emulate the forward model using the training set
    gp = gp_emulator.MultivariateEmulator( X = Obs_train - Obs_train_m, y=training_set, thresh=thresh)

    # # 4.0 Create validation dataset
    # Obs_validate = []
    # for xx in validate_set:
    #     theta=xx[3]
    #     output = do_cmem_fwd_model (xx[0:3], theta)
    #     hpol= float(output[0][0])
    #     vpol = float(output[1][0])
    #     Obs_validate.append([hpol,vpol])
    # Obs_validate = np.array ( Obs_validate )
    #
    # # 5.0 Predict validation dataset with emulator
    # Obs_emulated = []
    # for ii in xrange ( validate_set.shape[0]):
    #     Obs_emulated.append ( gp.predict(validate_set[ii])[0] + Obs_train_m)
    # Obs_emulated = np.array (Obs_emulated)
    #
    # # 6.0 Validate emulator by cross comparison statistics
    # # Error
    # err = Obs_emulated - Obs_validate
    #
    # # RMSE
    # #AE = err.__abs__()
    # SE = np.square(err)
    # MSE = np.mean(SE,0)
    # RMSE = np.sqrt(MSE)
    #
    # # mean error
    # ME = np.mean(err,0)
    #
    # #Maximum Absolute Error
    # #MAE = np.max(AE)
    #
    # # Quantiles Errors
    # qua = np.percentile ( err, [5,25,50,75,95], axis=0 )
    # AE = err.__abs__()
    # MSA = np.max(AE)
    #
    #
    # # 7.0 Plot differences
    # #import matplotlib.pyplot as plt
    # string = 'Ntrain=%02d, Thresh=%04.3f, MSA=%6.4f.png' %(ntrain, thresh, MSA)
    # filename1 = 'CreateEmulator_Quantiles_(Ntrain_%02d)_(Thresh_%04.3f).png' %(ntrain, thresh)
    # filename2 = 'CreateEmulator_RMSE_(Ntrain_%02d)_(Thresh_%04.3f).png' %(ntrain, thresh)
    #
    # plt.figure()
    # plt.fill_between ( wl, qua[0] ,qua[4], color="0.8")
    # plt.fill_between ( wl, qua[1] ,qua[3], color="0.6")
    # plt.plot ( wl, err.mean(axis=0), '-r')
    # plt.plot ( wl, qua[2], '--g')
    # plt.axhline(0, c="y")
    # plt.axhline(-0.01, c="y")
    # plt.axhline(+0.01, c="y")
    # plt.ylim(-0.05,0.05)
    # #plt.plot(wv, mu_x )
    # plt.xlim(400,2500)
    # plt.title(string)
    #
    # plt.savefig(filename1)
    # plt.close()
    #
    # plt.figure()
    # plt.semilogy ( wl, RMSE, '-r')
    # plt.semilogy ( wl, ME, '-b')
    # plt.semilogy ( wl, -ME, '--b')
    # plt.axhline(+0.01, c="y")
    # plt.ylabel('RMSE [-]')
    # plt.xlim(400,2500)
    # plt.axhline(+0.01, c="y")
    # plt.ylim(1e-5,1e-0)
    # plt.legend(['RMSE', 'Mean Error', '-Mean Error','Upper Thresh'])
    #
    # plt.savefig(filename2)
    # plt.close()

    # import pdb
    # pdb.set_trace()
    return gp

def create_prosail_emulators ( sza, vza, raa,ntrain=200,thresh=0.97):
    parameters = [ 'n', 'cab', 'car', 'cbrown', 'cw', 'cm', 'lai', 'ala', 'bsoil', 'psoil']
    min_vals = [ 0.8       ,  0.46301307,  0.95122942,  0.        ,  0.02829699,
                0.03651617,  0.04978707,  0.44444444,  0.        ,  0.]
    max_vals = [ 2.5       ,  0.998002  ,  1.        ,  1.        ,  0.80654144,
                0.84366482,  0.99501248,  0.55555556,  2.   , 1     ]
    training_set, distributions = gp_emulator.create_training_set ( parameters, min_vals, max_vals, n_train=ntrain )

    rho_train = []
    for xx in training_set:
        rho_train.append ( do_prosail_fwd_model ( xx, sza, vza, raa ) )
    rho_train = np.array ( rho_train )

    validate_set = gp_emulator.create_validation_set( distributions )
    rho_validate = []
    for xx in validate_set:
        rho_validate.append ( do_prosail_fwd_model ( xx, sza, vza, raa ) )
    rho_validate = np.array ( rho_validate )

    gp = gp_emulator.MultivariateEmulator( X=rho_train, y=training_set, thresh=thresh )
    return gp

def create_scope_emulators ( sza, vza, raa,ntrain=400,thresh=0.9999):
    print 'creating SCOPE emulators'
    option = 'BrightnessT'  #'Radiance'


    # # in short
    # from SCOPE_tools import SCOPE
    # scope_model                                 =   SCOPE()
    # Input_transformed                                       =   scope_model.Input_transformed
    #
    # input_values_transformed                    =   [value for name,value in Input_transformed.iteritems() if not(name=='tto' or  name =='tts' or name=='psi' or name =='M1' or name =='M2' or name =='M3' or name =='M4' or name =='M5' or name =='M6')]
    # names                                       =   [name for name,value in Input_transformed.iteritems() if not(name=='tto' or  name =='tts' or name=='psi' or name =='M1' or name =='M2' or name =='M3' or name =='M4' or name =='M5' or name =='M6')]
    # input_dict                                  =   zip(names, input_values_transformed)
    # ntrain = 200; sza = 10; vza=20; raa=30
    # scope_model.run_model( input_values_transformed, sza, vza, raa )


    # maybe add soil/leaf emissivities?
    from SCOPE_tools import SCOPE
    import numpy as np

    ###########################################################################################
    # initialize Model
    ###########################################################################################
    print '-Initializing SCOPE'
    scope_model                                 =   SCOPE()
    do_scope_fwd_model                          =   scope_model.run_model

    # specify height of vegetation (either 2 or 50 m).
    scope_model.Input_transformed['hc']         =   50.
    # specify reference heights according to ECMWF (defined as negative to be relative to vegetation height)
    scope_model.matlab.put('zm',-10.)
    scope_model.matlab.put('zh',-2.)
    scope_model.matlab.eval('SCOPE_translate_bck')

    Input_transformed                           =   scope_model.Input_transformed

    # outputdirectory                             =   '/home/ucfajti/Simulations/Python/scope/Output/Emulation_part_Vcmo/'

    # Define global dataset of parameters used in scope
    input_parameters                            = [name  for name,value in Input_transformed.iteritems() if not(name=='tto' or  name =='tts' or name=='psi')]
    input_values_transformed                    = [value for name,value in Input_transformed.iteritems() if not(name=='tto' or  name =='tts' or name=='psi')]
    input_dict_transformed                      = dict(zip(input_parameters,input_values_transformed))

    # input_parameters                            = ['Ta', 'p',           'ea',  'u', 'Rin',               'Rli', 'Cab', 'Cca','Cdm', 'Cw', 'Cs', 'N', 'emissivity_leaf', 'emissivity_soil', 'Vcmo', 'lai', 'hc', 'lw'] #, 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'tts','tto','psi'
    # input_values_transformed                    = [22.05,993.8/1.0e3,  67.8,  2.3,  440./1.0e3,  381.69/1.0e3,  0.89,  0.99, 0.38, 0.78, 0.60, 1.6,         0.98,                    0.95,    30., 0.95,  49.55,0.054] #transformed
    # input_dict_transformed                      = dict(zip(input_parameters,input_values_transformed))

    # the sensitivity analysis has proven that specific parameters impact the thermal radiation only minimally (in respect to other parameters). These parameters []
    # will be omitted in the emulation (to reduce the number of training samples required for a proper training), we should still add rho_tl and tau_tl
    # parameters                                  = ['Ta',   'u',   'Rin',       'Rli',        'lai', 'Vcmo']
    # min_vals_transformed                        = [0.,      0.01,  50./1.0e3,  100.00/1.0e3, 0.049,  8]
    # max_vals_transformed                        = [50.,     5.00, 600./1.0e3,  500.00/1.0e3, 0.99,  80]

    # # the sensitivity analysis has proven that specific parameters impact the thermal radiation only minimally (in respect to other parameters). These parameters []
    # will be omitted in the emulation (to reduce the number of training samples required for a proper training), we should still add rho_tl and tau_tl
    parameters                                          =   ['Ta',   'u',   'Rin',       'Rli',        'lai', 'emissivity_leaf', 'emissivity_soil', 'Vcmo']
    min_vals_transformed                                =   [0.,      0.01,  01./1.0e3,  100.00/1.0e3, 0.049,               0.85,            0.80,      8]
    max_vals_transformed                                =   [50.,    15.00, 900./1.0e3,  500.00/1.0e3, 0.99,                0.99,            0.99,     80]


    # check
    # print '-Perform SCOPE test run'
    # output                                              =   scope_model.run_model_with_dict(input_dict_transformed)
    # wl_MODIS                                            =   np.array (output['Bandwl'])
    # Bandnr                                              =   np.array( output['Bandnr']/1000)
    # ithm                                                =   np.any([Bandnr==31,Bandnr==32],axis=0)                   # if we would like to only band 31 and 32
    # print Bandnr, ithm



    # Setup how parameters in the subset-database are sorted into the global database
    print '-Create sorting to global database'
    Isort                                       =   np.zeros_like(min_vals_transformed).astype('int')
    for i1,name1 in enumerate(parameters):
        for i2,name2 in enumerate(input_parameters):
            if name1==name2:
                Isort[i1]                       =   i2


    ###########################################################################################
    # Train emulators
    ###########################################################################################
    import gp_emulator

    # ntrain = 200; sza = 10; vza=20; raa=30
    # Create trainingset
    training_set, distributions                     =   gp_emulator.create_training_set ( parameters, min_vals_transformed, max_vals_transformed, n_train=ntrain )

    #run model
    import copy
    print 'Run the SCOPE model for %03.0f scenarios, please wait..' % ntrain
    Output                                          =   []
    for i,xx_s in enumerate(training_set):
        # print float(i)/float(ntrain)*100.
        # update parameters of global dataset with values from subset
        xx_transformed                              =   copy.deepcopy(input_values_transformed)
        for i1,i2 in enumerate(Isort):
            xx_transformed[i2]                      =   xx_s[i1]
        # run the model and store the output
        Output.append ( scope_model.run_model( xx_transformed, sza, vza, raa ))


    # Define how output is handled
    if option=='Radiance':
        varname                                     =   'Lot_'
        unit                                        =   '[W/m2/um]'
        wl                                          =   wl
        ithm                                        =   wl>4.5e3

        wlT                                         =   wl[ithm]
        Varvalue_                                   =   np.array ( [output[varname] for output in Output])[:,ithm]

    elif option=='BrightnessT':
        varname                                     =   'Tb'
        unit                                        =   '[C]'
        wl_MODIS                                    =   np.array ( [output['Bandwl'] for output in Output])
        Tb_MODIS                                    =   np.array ( [output[varname] for output in Output])
        Bandnr                                      =   np.array(   output['Bandnr']/1000)

        # identify bands
        # ithm                                        =   wl_MODIS[0,:]>0                                         # if we would like to see all thermal bands
        ithm                                        =   np.any([Bandnr==31,Bandnr==32],axis=0)                   # if we would like to only band 31 and 32

        # select bands
        wlT                                         =   wl_MODIS[0,ithm]
        Varvalue_                                   =   Tb_MODIS[:,ithm]



    # create emulator
    print 'Create the SCOPE emulator (using the the different scenarios), please wait'
    gp_thermal                                      = gp_emulator.MultivariateEmulator( X=Varvalue_, y=training_set, thresh=thresh, n_tries=10)

    # it might be better to use band_emulators. in that case one does not need to specify thresholds (this is only required when
    # trying to represent specific spectral-features. Use the following code:
    # from functools import partial
    # retval = []
    # for iband,bmin in enumerate ( b_min ):
    #     print "Doing band %d" % (iband+1)
    #     passband = np.nonzero( np.logical_and ( wv >= bmin, wv <= b_max[iband] ) )
    #     simulator = partial ( rt_model, passband=passband )
    #     x = gp_emulator.create_emulator_validation ( simulator, parameters, min_vals, max_vals,
    #                                 n_train, n_validate, do_gradient=True,
    #                                 n_tries=15, args=(30, 0, 0) )
    #     retval.append ( x )

    ###########################################################################################
    # Validate emulators
    ###########################################################################################
    # # Reinitialize SCOPE (in case the training took to long and the matlab-session was closed)
    # scope_model                                 =   SCOPE()
    # do_scope_fwd_model                          =   scope_model.run_model
    # scope_model.matlab.put('zm',-10.)
    # scope_model.matlab.put('zh',-2.)
    #
    # # Create validation
    # validate_set                                = gp_emulator.create_validation_set( distributions )
    #
    #
    # # Run Sceanrios (training/validation)
    # nvalidate,nwl                               =    validate_set.shape
    # V_validate                                  =   np.zeros([len(validate_set), len(wlT)])
    # V_emulate                                   =   np.zeros([len(validate_set), len(wlT)])
    #
    # for i,xx_transformed_s in enumerate(validate_set):
    #     print float(i)/float(ntrain)*100.
    #     xx_transformed                          =   copy.deepcopy(input_values_transformed)
    #
    #     # update parameters of global dataset with values from subset
    #     for i1,i2 in enumerate(Isort):
    #         xx_transformed[i2]                  = xx_transformed_s[i1]
    #
    #     # run the model and store the output
    #     output                                  =   do_scope_fwd_model ( xx, sza, vza, raa )
    #     V_validate[i,:]                         =   output[varname][ithm]
    #
    #     # run emulated model to created emulated set
    #     v_emulate,drho_                         =   gp_thermal.predict(xx_s)
    #     V_emulate[i,:]                          =   v_emulate
    # print 100.
    ###########################################################################################

    return gp_thermal


def do_cmem_fwd_model ( values, theta):
    import cmem
    nbpt_i              =    30942
    ndate_i             =    30942
    ntime_i             =    30942
    doy_i               =    195.25

    # vza                 =   0
    # raa                 =   0

    fghz_i              =   1.4

    # soil type
    fsand_i             =   44.81640625
    fclay_i             =   21.92578125
    fZ_i                =   1852.44335938

    # vegetation type
    fTVL_i              =   1
    fTVH_i              =   19
    fvegl_i             =   0.708160400391
    fvegh_i             =   0.291839599609
    fwater_i            =   0.0003662109375

    ftskin_i            =   291.427
    fsnowd_i            =   1.48225e-08
    frsnow_i            =   100

    zlaiH_i            =   3.49560546875
    ftair_i            =   287.5255

    fwc_lsm_i2         =   0.204851
    fwc_lsm_i3         =   0.273826

    ftl_lsm_i2         =   290.1147
    ftl_lsm_i3         =   values[1]

    x                   =   cmem.run_cmem(nbpt_i,ndate_i,ntime_i, doy_i,
                                           fghz_i, theta,
                                          [values[ 0], fwc_lsm_i2, fwc_lsm_i3],
                                          [values[ 1], ftl_lsm_i2, ftl_lsm_i3],
                                           ftskin_i, fsnowd_i, frsnow_i,
                                           fsand_i, fclay_i, fZ_i,
                                           fTVL_i, fTVH_i, fvegl_i, fvegh_i, fwater_i,
                                           values[ 2], zlaiH_i, ftair_i)

    return x

def do_prosail_fwd_model ( x, sza, vza, raa ):
    x = inverse_transform ( x )
    ################# surface refl with prosail #####################
    surf_refl = prosail.run_prosail(x[0], x[1], x[2], x[3], \
        x[4], x[5], x[6], x[7], 0, x[8], x[9], 0.01, sza, vza, raa, 2 )
    return surf_refl


def inverse_transform ( x ):
    """Inverse transform the PROSAIL parameters"""
    x_out = x*1.
    # Cab, posn 1
    x_out[1] = -100.*np.log ( x[1] )
    # Car, posn 2
    x_out[2] = -100.*np.log ( x[2] )
    # Cw, posn 4
    x_out[4] = (-1./50.)*np.log ( x[4] )
    #Cm, posn 5
    x_out[5] = (-1./100.)*np.log ( x[5] )
    # LAI, posn 6
    x_out[6] = -2.*np.log ( x[6] )
    # ALA, posn 7
    x_out[7] = 90.*x[7]
    return x_out





class StandardStateSCOPESAIL ( State ):
    """A standard state configuration for the CMEM model required for eoldas"""
    def __init__ ( self, state_config, state_grid, optimisation_options=None, output_name=None, verbose=False ):
        self.state_config                       =   state_config
        self.state_grid                         =   state_grid
        self.n_elems                            =   self.state_grid.size
        self.verbose                            =   verbose

        # Define the default values of SCOPE (thermal) (untransformed)
        self.default_values                     =   OrderedDict ()
        self.default_values['Ta']               =   20.
        self.default_values['u']                =   2.
        self.default_values['Rin']              =   450.
        self.default_values['Rli']              =   250.
        self.default_values['lai']              =   2.89106446505
        self.default_values['emissivity_leaf']  =   0.98
        self.default_values['emissivity_soil']  =   0.95
        self.default_values['Vcmo']             =   30.

        self.parameter_min                      =   OrderedDict()
        self.parameter_min['Ta']                =   -10.
        self.parameter_min['u']                 =   1.e-2
        self.parameter_min['Rin']               =   1.e-6
        self.parameter_min['Rli']               =   1.e-6
        self.parameter_min['lai']               =   0.00000
        self.parameter_min['emissivity_leaf']   =   0.85
        self.parameter_min['emissivity_soil']   =   0.80
        self.parameter_min['Vcmo']              =   8.

        self.parameter_max                      =   OrderedDict()
        self.parameter_max['Ta']                =   50.
        self.parameter_max['u']                 =   15.
        self.parameter_max['Rin']               =   850.
        self.parameter_max['Rli']               =   500.
        self.parameter_max['lai']               =   8.00000 #15.00000
        self.parameter_max['emissivity_leaf']   =   0.99
        self.parameter_max['emissivity_soil']   =   0.99
        self.parameter_max['Vcmo']              =   80.

        # Define the default values of PROSAIL (untransformed)
        # according to Malenovsky et al, 2007, Applicability of the PROSPECT model for Norway spruce needles (Int. jour. RS)
        self.default_values['n']                =   2.0 #1.6
        self.default_values['cab']              =   70.
        self.default_values['car']              =   10.
        self.default_values['cbrown']           =   0.01
        self.default_values['cw']               =   0.05    # Say?
        self.default_values['cm']               =   0.025  # Say?
        # self.default_values['lai']              =   2
        self.default_values['ala']              =   70.
        self.default_values['bsoil']            =   0.5
        self.default_values['psoil']            =   0.7


        self.parameter_min['n']                 =   0.8
        self.parameter_min['cab']               =   0.2
        self.parameter_min['car']               =   0.0
        self.parameter_min['cbrown']            =   0.0
        self.parameter_min['cw']                =   0.0043
        self.parameter_min['cm']                =   0.0017
        self.parameter_min['ala']               =   0.0
        self.parameter_min['bsoil']             =   -1.0
        self.parameter_min['psoil']             =   -1.0

        self.parameter_max['n']                 =   2.5
        self.parameter_max['cab']               =   77.0
        self.parameter_max['car']               =   5.0
        self.parameter_max['cbrown']            =   1.0
        self.parameter_max['cw']                =   0.0753
        self.parameter_max['cm']                =   0.0331 # Say?
        # self.parameter_max['lai']              =   10
        self.parameter_max['ala']               =   90.
        self.parameter_max['bsoil']             =   2.0
        self.parameter_max['psoil']             =   1.0

        self.operators                          =   {}
        self.n_params                           =   self._state_vector_size ()


        # HACK
        self.transformation_dict = {}
        self.invtransformation_dict = {}

        self.bounds                             =   []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            self.bounds.append ( [ self.parameter_min[param]*(1-1e-4), self.parameter_max[param]*(1+1e-4) ] )

        if output_name is None:
            tag                                 =   time.strftime( "%04Y%02m%02d_%02H%02M%02S_", time.localtime())
            tag                                 =   tag+platform.node()
            self.output_name                    =   "eoldas_retval_%s.pkl" % tag
        else:
            self.output_name                    =   output_name

        print "Saving results to %s" % self.output_name
        if optimisation_options is None:
            self.optimisation_options           =   {"factr": 1000, "m":400, "pgtol":1e-12, "maxcor":200,
                                                     "maxiter":10000, "disp":True }
            self.optimisation_options           =   {"factr": 1000, "m":400, "pgtol":1e-12, "maxcor":200,
                                                     "maxiter":3000, "disp":True }
        else:
            self.optimisation_options           =   optimisation_options

        # The following things are required when using Netcdf to store intermediate results
        self.netcdf                             = False
        self.metadata = MetaState()
        self.metadata.add_variable ( "n","None", "PROSPECT leaf layers", "leaf_layers" )
        self.metadata.add_variable ( "cab","microgram per centimetre^2","PROSPECT leaf chlorophyll content","cab" )
        self.metadata.add_variable ( "car","microgram per centimetre^2","PROSPECT leaf carotenoid content","car" )
        self.metadata.add_variable ( "cbrown","fraction", "PROSPECT leaf senescent fraction","cbrown" )
        self.metadata.add_variable ( "cw","centimetre", "PROSPECT equivalent leaf water","cw" )
        self.metadata.add_variable ( "cm","gram per centimeter^2", "PROSPECT leaf dry matter","cm" )
        self.metadata.add_variable ( "lai","meter^2 per meter^2", "Leaf Area Index","lai" )
        self.metadata.add_variable ( "ala","degree", "Average leaf angle","ala" )
        self.metadata.add_variable ( "bsoil","", "Soil brightness","bsoil" )
        self.metadata.add_variable ( "psoil","", "Soil moisture term","psoil" )

class StandardStateCMEMSAIL ( State ):
    """A standard state configuration for the CMEM model required for eoldas"""
    def __init__ ( self, state_config, state_grid, optimisation_options=None, output_name=None, verbose=False ):
        self.state_config                       =   state_config
        self.state_grid                         =   state_grid
        self.n_elems                            =   self.state_grid.size
        self.verbose                            =   verbose

        # Define the default values of CMEM
        self.default_values                     =   OrderedDict ()
        self.default_values['fwc_lsm_i1']       =   0.214766
        self.default_values['ftl_lsm_i1']       =   290.8225
        self.default_values['lai']        =   2.89106446505

        self.parameter_min                      =   OrderedDict()
        self.parameter_min['fwc_lsm_i1']        =   0.00000
        self.parameter_min['ftl_lsm_i1']        =   225.0 #225.000
        self.parameter_min['lai']         =   0.00000

        self.parameter_max                      =   OrderedDict()
        self.parameter_max['fwc_lsm_i1']        =   1.00000
        self.parameter_max['ftl_lsm_i1']        =   350.000
        self.parameter_max['lai']         =   8.00000 #15.00000

        # Define the default values of SAIL
        self.default_values['n']                =   1.6
        self.default_values['cab']              =   20.
        self.default_values['car']              =   1.
        self.default_values['cbrown']           =   0.01
        self.default_values['cw']               =   0.018 # Say?
        self.default_values['cm']               =   0.03 # Say?
        # self.default_values['lai']              =   2
        self.default_values['ala']              =   70.
        self.default_values['bsoil']            =   0.5
        self.default_values['psoil']            =   0.9


        self.parameter_min['n']                 =   0.8
        self.parameter_min['cab']               =   0.2
        self.parameter_min['car']               =   0.0
        self.parameter_min['cbrown']            =   0.0
        self.parameter_min['cw']                =   0.0043
        self.parameter_min['cm']                =   0.0017
        # self.parameter_min['lai']              =   0.0001
        self.parameter_min['ala']               =   0.0
        self.parameter_min['bsoil']             =   -1.0
        self.parameter_min['psoil']             =   -1.0

        self.parameter_max['n']                 =   2.5
        self.parameter_max['cab']               =   77.0
        self.parameter_max['car']               =   5.0
        self.parameter_max['cbrown']            =   1.0
        self.parameter_max['cw']                =   0.0753
        self.parameter_max['cm']                =   0.0331 # Say?
        # self.parameter_max['lai']              =   10
        self.parameter_max['ala']               =   90.
        self.parameter_max['bsoil']             =   2.0
        self.parameter_max['psoil']             =   1.0

        self.operators                          =   {}
        self.n_params                           =   self._state_vector_size ()


        # HACK
        self.transformation_dict = {}
        self.invtransformation_dict = {}

        self.bounds                             =   []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            self.bounds.append ( [ self.parameter_min[param]*(1-1e-4), self.parameter_max[param]*(1+1e-4) ] )

        if output_name is None:
            tag                                 =   time.strftime( "%04Y%02m%02d_%02H%02M%02S_", time.localtime())
            tag                                 =   tag+platform.node()
            self.output_name                    =   "eoldas_retval_%s.pkl" % tag
        else:
            self.output_name                    =   output_name

        print "Saving results to %s" % self.output_name
        if optimisation_options is None:
            self.optimisation_options           =   {"factr": 1000, "m":400, "pgtol":1e-12, "maxcor":200,
                                                     "maxiter":10000, "disp":True }
            self.optimisation_options           =   {"factr": 1000, "m":400, "pgtol":1e-12, "maxcor":200,
                                                     "maxiter":3000, "disp":True }
        else:
            self.optimisation_options           =   optimisation_options

        # The following things are required when using Netcdf to store intermediate results
        self.netcdf                             = False
        self.metadata = MetaState()
        self.metadata.add_variable ( "n","None", "PROSPECT leaf layers", "leaf_layers" )
        self.metadata.add_variable ( "cab","microgram per centimetre^2","PROSPECT leaf chlorophyll content","cab" )
        self.metadata.add_variable ( "car","microgram per centimetre^2","PROSPECT leaf carotenoid content","car" )
        self.metadata.add_variable ( "cbrown","fraction", "PROSPECT leaf senescent fraction","cbrown" )
        self.metadata.add_variable ( "cw","centimetre", "PROSPECT equivalent leaf water","cw" )
        self.metadata.add_variable ( "cm","gram per centimeter^2", "PROSPECT leaf dry matter","cm" )
        self.metadata.add_variable ( "lai","meter^2 per meter^2", "Leaf Area Index","lai" )
        self.metadata.add_variable ( "ala","degree", "Average leaf angle","ala" )
        self.metadata.add_variable ( "bsoil","", "Soil brightness","bsoil" )
        self.metadata.add_variable ( "psoil","", "Soil moisture term","psoil" )

class StandardStateCMEM ( State ):
    """A standard state configuration for the CMEM model required for eoldas"""
    def __init__ ( self, state_config, state_grid, optimisation_options=None, output_name=None, verbose=False ):
        self.state_config                       =   state_config
        self.state_grid                         =   state_grid
        self.n_elems                            =   self.state_grid.size
        self.verbose                            =   verbose

        # Now define the default values
        self.default_values                     =   OrderedDict ()
        self.default_values['fwc_lsm_i1']       =   0.214766
        self.default_values['ftl_lsm_i1']       =   290.8225
        self.default_values['fs_laiL_i']        =   2.89106446505
        self.default_values['test']             =   0.0

        # self.parameter_min                      =   OrderedDict()
        # self.parameter_min['fwc_lsm_i1']        =   0.00000
        # self.parameter_min['ftl_lsm_i1']        =   275.000
        # self.parameter_min['fs_laiL_i']         =   0.00000

        self.parameter_min                      =   OrderedDict()
        self.parameter_min['fwc_lsm_i1']        =   0.00000
        self.parameter_min['ftl_lsm_i1']        =   225.0 #225.000
        self.parameter_min['fs_laiL_i']         =   0.00000
        self.parameter_min['test']              =   -1.0

        # self.parameter_max                      =   OrderedDict()
        # self.parameter_max['fwc_lsm_i1']        =   1.00000
        # self.parameter_max['ftl_lsm_i1']        =   315.000
        # self.parameter_max['fs_laiL_i']         =   8.00000

        self.parameter_max                      =   OrderedDict()
        self.parameter_max['fwc_lsm_i1']        =   1.00000
        self.parameter_max['ftl_lsm_i1']        =   350.000
        self.parameter_max['fs_laiL_i']         =   8.00000 #15.00000
        self.parameter_max['test']              =   1.0

        self.operators                          =   {}
        self.n_params                           =   self._state_vector_size ()


        # HACK
        self.transformation_dict = {}
        self.invtransformation_dict = {}

        self.bounds                             =   []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            self.bounds.append ( [ self.parameter_min[param]*(1-1e-4), self.parameter_max[param]*(1+1e-4) ] )

        if output_name is None:
            tag                                 =   time.strftime( "%04Y%02m%02d_%02H%02M%02S_", time.localtime())
            tag                                 =   tag+platform.node()
            self.output_name                    =   "eoldas_retval_%s.pkl" % tag
        else:
            self.output_name                    =   output_name

        print "Saving results to %s" % self.output_name
        if optimisation_options is None:
            self.optimisation_options           =   {"factr": 1000, "m":400, "pgtol":1e-12, "maxcor":200,
                                                     "maxiter":10000, "disp":True }
            self.optimisation_options           =   {"factr": 1000, "m":400, "pgtol":1e-12, "maxcor":200,
                                                     "maxiter":3000, "disp":True }
        else:
            self.optimisation_options           =   optimisation_options

        # The following things are required when using Netcdf to store intermediate results
        self.netcdf                             = False
        self.metadata = MetaState()
        self.metadata.add_variable ( "n","None", "PROSPECT leaf layers", "leaf_layers" )
        self.metadata.add_variable ( "cab","microgram per centimetre^2","PROSPECT leaf chlorophyll content","cab" )
        self.metadata.add_variable ( "car","microgram per centimetre^2","PROSPECT leaf carotenoid content","car" )
        self.metadata.add_variable ( "cbrown","fraction", "PROSPECT leaf senescent fraction","cbrown" )
        self.metadata.add_variable ( "cw","centimetre", "PROSPECT equivalent leaf water","cw" )
        self.metadata.add_variable ( "cm","gram per centimeter^2", "PROSPECT leaf dry matter","cm" )
        self.metadata.add_variable ( "lai","meter^2 per meter^2", "Leaf Area Index","lai" )
        self.metadata.add_variable ( "ala","degree", "Average leaf angle","ala" )
        self.metadata.add_variable ( "bsoil","", "Soil brightness","bsoil" )
        self.metadata.add_variable ( "psoil","", "Soil moisture term","psoil" )

class StandardStateCMEM2 ( State ):
    """A standard state configuration for the CMEM model required for eoldas"""
    def __init__ ( self, state_config, state_grid, optimisation_options=None, output_name=None, verbose=False ):


        self.state_config                       =   state_config
        self.state_grid                         =   state_grid
        self.n_elems                            =   self.state_grid.size
        self.verbose                            =   verbose

        # Now define the default values
        self.default_values                     =   OrderedDict ()
        self.default_values['fwc_lsm_i1']       =   0.214766
        self.default_values['ftl_lsm_i1']       =   290.8225
        self.default_values['fs_laiL_i']        =   2.89106446505
        self.default_values['theta']            =   40

        self.parameter_min                      =   OrderedDict()
        self.parameter_min['fwc_lsm_i1']        =   0.00000
        self.parameter_min['ftl_lsm_i1']        =   273.000
        self.parameter_min['fs_laiL_i']         =   0.00000
        self.parameter_min['theta']             =   0.00000

        self.parameter_max                      =   OrderedDict()
        self.parameter_max['fwc_lsm_i1']        =   1.00000
        self.parameter_max['ftl_lsm_i1']        =   325.000
        self.parameter_max['fs_laiL_i']         =   10.00000
        self.parameter_max['theta']             =   85.00000

        self.operators                          =   {}
        self.n_params                           =   self._state_vector_size ()

        # HACK
        self.transformation_dict = {}
        self.invtransformation_dict = {}

        self.bounds                             =   []
        for ( i, param ) in enumerate ( self.state_config.iterkeys() ):
            self.bounds.append ( [ self.parameter_min[param]*0.9, self.parameter_max[param]*1.1 ] )

        if output_name is None:
            tag                                 =   time.strftime( "%04Y%02m%02d_%02H%02M%02S_", time.localtime())
            tag                                 =   tag+platform.node()
            self.output_name                    =   "eoldas_retval2_%s.pkl" % tag
        else:
            self.output_name                    =   output_name

        print "Saving results to %s" % self.output_name
        if optimisation_options is None:
            self.optimisation_options           =   {"factr": 1000, "m":400, "pgtol":1e-12, "maxcor":200,
                                                     "maxiter":1500, "disp":True }
        else:
            self.optimisation_options           =   optimisation_options


class ObservationOperatorNew ( ObservationOperatorTimeSeriesGP ):
    """
    A practical class that changes how we "fish out" the emulators from
    ``self.emulators``. All we need to do is to add a new ``time_step``
    method to the class.
    """
    def __init__ ( self, state_grid, state, observations, mask, emulators, bu, \
            band_pass=None, bw=None ):
        ObservationOperatorTimeSeriesGP.__init__ ( self, state_grid, state, observations, \
                                                  mask, emulators, bu, band_pass, bw )
        self.sel_emu_keys = []
        keys = np.array(emulators.keys())

        emu_locs = [np.argmin(np.sum(( keys - mask[i, [3,2,4]])**2,axis=1))
                        for i in xrange(mask.shape[0]) ]

        self.sel_emu_keys = keys[emu_locs]

    def time_step ( self, this_loc ):
        k = tuple(self.sel_emu_keys[this_loc])
        this_obs = self.observations[ :, this_loc]
        if self.bu.ndim == 1:
            this_bu = self.bu
        elif self.bu.ndim == 2:
            this_bu = self.bu[:, this_loc ]

        return self.emulators[k], this_obs, this_bu, [self.band_pass, self.bw]


#test

if __name__ == "__main__":
    ###########################################################################################
    # investigate how scope can be best emulated (using subsets of parameters instead of the full global parameter-set)
    ###########################################################################################
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    # from datetime import datetime as datet
    import datetime
    from eoldas_ng import *
    ###########################################################################################
    print 'creating SCOPE emulators'

    option                                                      = 'Radiance'
    option                                                      = 'BrightnessT'
    sza                                                         =   20.
    vza                                                         =   30.
    raa                                                         =   0.

    threshs                                                     =  [1-1e-3, 1-1e-4, 1-1e-5, 1-1e-6]
    threshs                                                     =  [1-1e-6]
    ntrains                                                     =  [400] #[50, 100, 200, 400]


    for ntrain in ntrains:
        for thresh in threshs:
            ###########################################################################################
            # setup scope
            ###########################################################################################

            from SCOPE_tools import SCOPE
            print '-Initializing SCOPE'
            plt.ion()
            scope_model                                         =   SCOPE()
            Input_transformed                                   =   scope_model.Input_transformed
            wl                                                  =   scope_model.wl
            do_scope_fwd_model                                  =   scope_model.run_model


            # specify reference heights according to ECMWF (defined as negative to be relative to vegetation height)
            scope_model.matlab.put('zm',-10.)
            scope_model.matlab.put('zh',-2.)

            input_parameters                                    =   ['Ta', 'p',           'ea',  'u', 'Rin',               'Rli', 'Cab', 'Cca','Cdm', 'Cw', 'Cs', 'N', 'emissivity_leaf', 'emissivity_soil', 'Vcmo', 'lai', 'hc', 'lw'] #, 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'tts','tto','psi'
            input_values_transformed                            =   [22.05,993.8/1.0e3,  67.8,  2.3,  440./1.0e3,  381.69/1.0e3,  0.89,  0.99, 0.38, 0.78, 0.60, 1.6,               0.95,              0.90,    30.,  0.95,  49.55,0.054] #transformed
            input_dict_transformed                              =   dict(zip(input_parameters,input_values_transformed))

            # # the sensitivity analysis has proven that specific parameters impact the thermal radiation only minimally (in respect to other parameters). These parameters []
            # will be omitted in the emulation (to reduce the number of training samples required for a proper training), we should still add rho_tl and tau_tl
            parameters                                          =   ['Ta',   'u',   'Rin',       'Rli',        'lai', 'emissivity_leaf', 'emissivity_soil', 'Vcmo']
            min_vals_transformed                                =   [0.,      0.01,  50./1.0e3,  100.00/1.0e3, 0.049,               0.85,            0.80,      8]
            max_vals_transformed                                =   [50.,     5.00, 600./1.0e3,  500.00/1.0e3, 0.99,                0.99,            0.99,     80]
            outputdirectory                                     =   '/home/ucfajti/Simulations/Python/scope/Output/Emulation_part_Vcmo/'


            # check
            # print '-Perform SCOPE test run'
            # output                                              =   scope_model.run_model_with_dict(input_dict_transformed)
            # wl_MODIS                                            =   np.array (output['Bandwl'])
            # Bandnr                                              =   np.array( output['Bandnr']/1000)
            # ithm                                                =   np.any([Bandnr==31,Bandnr==32],axis=0)                   # if we would like to only band 31 and 32



            # Setup how parameters in the subset-database are sorted into the global database
            print '-Create sorting to global database'
            Isort                                               =   np.zeros_like(min_vals_transformed).astype('int')
            for i1,name1 in enumerate(parameters):
                for i2,name2 in enumerate(input_parameters):
                    if name1==name2:
                        Isort[i1]                               =   i2


            ###########################################################################################
            # create_scope_emulators ( sza, vza, raa,ntrain=ntrain,thresh=thresh)
            ###########################################################################################
            # Perform training-validate process for 5 times (to reduce noise)
            print 'Run Different Scenarios'
            gps                                                 =   []
            c=[]
            for igp in xrange(5):
                # Create trainingset
                training_set, distributions                     =   gp_emulator.create_training_set ( parameters, min_vals_transformed, max_vals_transformed, n_train=ntrain )

                # Run Scenarios (training/validation)
                Output                                          =   []
                for i,xx_transformed_s in enumerate(training_set):
                    xx_transformed                              =   copy.deepcopy(input_values_transformed)
                    for i1,i2 in enumerate(Isort):
                        xx_transformed[i2]                      =   xx_transformed_s[i1]


                    q
                    Output.append ( scope_model.run_model( xx_transformed, sza, vza, raa ))
                    print float(i)/float(ntrain)*100.
                print 100.

                ###########################################################################################
                # extract output (radiance or brightness temperatures)
                ###########################################################################################
                if option=='Radiance':
                    varname                                     =   'Lot_'
                    unit                                        =   '[W/m2/um]'
                    wl                                          =   wl
                    ithm                                        =   wl>4.5e3

                    wlT                                         =   wl[ithm]
                    Varvalue_                                   =   np.array ( [output[varname] for output in Output])[:,ithm]

                elif option=='BrightnessT':

                    varname                                     =   'Tb'
                    unit                                        =   '[C]'
                    wl_MODIS                                    =   np.array ( [output['Bandwl'] for output in Output])
                    Bandnr                                      =   np.array(   output['Bandnr']/1000)
                    Tb_MODIS                                    =   np.array ( [output[varname] for output in Output])


                    # identify bands
                    ithm                                        =   wl_MODIS[0,:]>0                                         # if we would like to see all thermal bands
                    # ithm                                        =   np.any([Bandnr==31,Bandnr==32],axis=0)                   # if we would like to only band 31 and 32

                    # select bands
                    wlT                                         =   wl_MODIS[0,ithm]
                    Varvalue_                                   =   Tb_MODIS[:,ithm]

                print 'create thermal emulator'

                gp_thermal                                      =   gp_emulator.MultivariateEmulator( X=Varvalue_, y=training_set, thresh=thresh, n_tries=10)
                q
                gps.append(gp_thermal)


            # ###########################################################################################
            # validate scope thermal emulators (using all parameters) ( sza, vza, raa,ntrain=ntrain,thresh=thresh)

            print 'Validate scope '+ option +'-emulators '
            validate_set                                        =   gp_emulator.create_validation_set( distributions )
            # if ntrain<500:

            # Reinitialize SCOPE
            scope_model                                         =   SCOPE()
            do_scope_fwd_model                                  =   scope_model.run_model
            scope_model.matlab.put('zm',-10.)
            scope_model.matlab.put('zh',-2.)

            # Run Scenarios (training/validation)
            nvalidate,nwl                                       =   validate_set.shape
            V_validate                                          =   np.zeros([len(validate_set), len(gps), len(wlT)])
            V_emulate                                           =   np.zeros([len(validate_set), len(gps), len(wlT)])
            for i,xx_transformed_s in enumerate(validate_set):
                xx_transformed                                  =   copy.deepcopy(input_values_transformed)
                for i1,i2 in enumerate(Isort):
                    xx_transformed[i2]                          =   xx_transformed_s[i1]


                # run real model to create validation set
                output                                          =   do_scope_fwd_model ( xx_transformed, sza, vza, raa )
                for igps,gp_thermal in enumerate(gps):
                    V_validate[i,igps,:]                        =   output[varname][ithm]

                # run emulated model to created emulated set
                for igps,gp_thermal in enumerate(gps):
                    v_emulate,drho_                             =   gp_thermal.predict(xx_transformed_s)
                    V_emulate[i,igps,:]                         =   v_emulate
                print float(i)/float(len(validate_set))*100.
            print 100.





            ################################################################################################################################
            # investigate errors-statistics (quantiles)
            ################################################################################################################################
            # calculate errors
            error_abs                                           =   (V_validate-V_emulate)
            error_rel                                           =   error_abs/V_validate * 100          # use relative error

            if option=='Radiance':
                error                                           =   error_rel                           # use abs error
                ylabel_str                                      =   'Rel '+option+' Errors [%]'
            elif option=='BrightnessT':
                error                                           =   error_abs                           # use abs error (goal is <1.0K)
                ylabel_str                                      =   '$\Delta$'+option+' '+ unit

            # calculate statistics
            error_mean                                          =   np.mean(error,0)                    # average over all validation trainingsets
            error_q05                                           =   np.percentile(error, 5,0)
            error_q25                                           =   np.percentile(error,25,0)
            error_q75                                           =   np.percentile(error,75,0)
            error_q95                                           =   np.percentile(error,95,0)

            i1                                                  =   wlT> 7500
            i2                                                  =   wlT<15000
            iwlt                                                =   i1&i2

            mean_error_q25                                      =   np.mean([np.mean(np.abs(error_q25[:,iwlt])), np.mean(error_q75[:,iwlt])])
            max_error_q25                                       =   np.max([np.max(np.abs(error_q25[:,iwlt])), np.max(error_q75[:,iwlt])])
            min_error_q25                                       =   np.min([np.min(np.abs(error_q25[:,iwlt])), np.min(error_q75[:,iwlt])])

            # Plot output
            simstr                                              =   'Errors@[7.5 - 15.0 um]: (ntrain = %2.0f' % ntrain + ', thresh = %5.4e)' % thresh
            errorstr                                            =   'Mean = %3.2f, ' % np.mean(error_mean[:,iwlt]) + ' Q25-Q50 [min - max] = [%3.2f' % min_error_q25 + ' - %3.2f]' %max_error_q25
            filestr                                             = 'Validation of SCOPE '+option+' emulator (thermal part)(Ntrain = %03.0f' % ntrain + ', thresh %e' % thresh +') - differences.png'

            plt.figure(figsize=[20, 20])
            for igps,gp in enumerate(gps):
                plt.fill_between(wlT,error_q05[igps,:],error_q95[igps,:],color='black', alpha=0.1)
            for igps,gp in enumerate(gps):
                plt.plot(wlT,error_mean[igps,:],'g',linewidth=2)
                plt.fill_between(wlT,error_q25[igps,:],error_q75[igps,:],color='red', alpha=0.4)
            plt.vlines( 7500,np.min(error_q05), np.max(error_q95) )
            plt.vlines(15000,np.min(error_q05), np.max(error_q95) )

            plt.xlabel('wavelength [nm]')
            plt.ylabel(ylabel_str)
            plt.title(simstr + ', ' + errorstr)
            plt.savefig(outputdirectory+filestr)


            ################################################################################################################################
            # investigate emulation (of training 1 and 2) in detail
            ################################################################################################################################
            # color = ['b','r','g','m','c','y','k','w']
            # marker = ['-o','->']
            #
            # Nr_runs = np.shape(V_validate)[1]
            # it = list(np.linspace(0,1,np.min([Nr_runs,2])).astype('int'));
            #
            # plt.figure(figsize=[15,5]);
            # plt.subplot(2,1,1);
            # plt.title('Absolute Values')
            # legendstr                                   =   []
            # for ii in it:
            #     plt.plot(wlT,V_validate[0,ii,:].T,'g'+marker[ii])
            #     plt.plot(wlT,V_emulate[0,ii,:].T,'r'+marker[ii])
            #     legendstr.append('training %03.0f' % ii)
            #     legendstr.append('emulate %03.0f' % ii)
            # plt.legend(legendstr)
            # plt.ylabel(option  +' '+ unit)
            # plt.subplot(2,1,2);
            # plt.title('Absolute Errors')
            # for ii in it:
            #     plt.plot(wlT,error_abs[0,it[ii],:].T,color[np.min([ii,7])])
            # filestr = 'Validation of SCOPE '+option+' emulator (thermal part)(Ntrain = %03.0f,' % ntrain + 'thresh=%e' % thresh + ') -Training 1 and 2 - example of Values.png'
            # plt.legend(['Training1','Training2','Training3'])
            # plt.ylabel('$\Delta$'+option +' '+ unit)
            # plt.xlabel('wavelength [nm]')
            # plt.savefig(outputdirectory+filestr)
            #
            # plt.figure(figsize=[15,5]);
            # plt.subplot(2,1,1);
            # legendstr                                   =   []
            # for i in xrange(np.min([Nr_runs,len(it)])):
            #     basisfunctions                          =   gps[it[i]].basis_functions
            #     Nr_basisfunc                            =   np.shape(basisfunctions)[0]
            #     for ibf in xrange(Nr_basisfunc):
            #         plt.plot(wlT,np.abs(basisfunctions[ibf,:]),color[np.min([ibf,7])]+marker[i])
            #         legendstr.append('BF%01.0f' % ibf + '_Train%01.0f'%i)
            # plt.legend(legendstr)
            # plt.title('Basis functions of the emulators')
            #
            # plt.subplot(2,1,2);
            # legendstr                                   =   []
            # for i in xrange(np.min([Nr_runs,len(it)])):
            #     hyperparams                             =   gps[it[i]].hyperparams
            #     Nr_hyperparams                          =   np.shape(hyperparams)[1]
            #     for ihp in xrange(Nr_hyperparams):
            #         plt.plot(hyperparams[:,ihp],color[np.min([ihp,7])]+'-'+marker[i]);
            #         legendstr.append('HP%01.0f' % ihp + '_Train%01.0f'%i)
            # plt.xticks(np.arange(len(parameters)), parameters);
            # plt.legend(legendstr)
            # plt.title('Hyperparameters of the emulators')
            # filestr = 'Validation of SCOPE '+option+' emulator (thermal part)(Ntrain = %03.0f,' % ntrain + 'thresh=%e' % thresh + ') -Training 1 and 2 - example of Basisfunctions and  Hyperparameters.png'
            # plt.savefig(outputdirectory+filestr)
            ################################################################################################################################
            plt.close('all')
