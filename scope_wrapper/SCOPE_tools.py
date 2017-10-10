
import matlab_wrapper
import numpy as np
import copy
from collections import OrderedDict
# matlab = matlab_wrapper.MatlabSession(matlab_root='/usr/local/matlab_r2014a/',options='-nojvm -nosplash')

import socket
hostname                                                =   socket.gethostname()
if hostname=='hephaestus.geog.ucl.ac.uk':
    matlab_root='/usr/local/matlab_r2014a/'
else:
    matlab_root='/usr/local/matlab_r2016a/'

matlab = matlab_wrapper.MatlabSession(matlab_root=matlab_root,options='-nojvm -nosplash')


## SCOPE.m (script)
#     SCOPE is a coupled radiative transfer and energy balance model
#     Copyright (C) 2015  Christiaan van der Tol
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY') without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

def Read_GlobCover(lon_s=05.2913,lat_s = 52.1326, Nx=1, Ny=1):
    import gdal
    file_LC         = '/home/ucfajti/Data/Satellites/GlobCover/V2.3/GLOBCOVER_L4_200901_200912_V2.3.tif'

    # if more pixels are required, expand the area (as globcover has a 3times higher resolution
    if Nx<>1:
        Nx = Nx*3

    if Ny<>1:
        Ny = Ny*3

     # open file
    ds          =   gdal.Open(file_LC)

    # determine coordinates
    Nlon        =   ds.RasterXSize
    Nlat        =   ds.RasterYSize
    gt          =   ds.GetGeoTransform()

    minlon      =   gt[0]
    maxlon      =   gt[0] + Nlon*gt[1] + Nlat*gt[2]
    minlat      =   gt[3]
    maxlat      =   gt[3] + Nlon*gt[4] + Nlat*gt[5]

    lon         =   np.linspace(minlon,maxlon,Nlon)
    lat         =   np.linspace(minlat,maxlat,Nlat)

    # Selecct pixel(s)
    difflon     =   np.abs(lon-lon_s)
    difflat     =   np.abs(lat-lat_s)

    ilon        =   np.where(min(difflon)==difflon)[0][0]
    ilat        =   np.where(min(difflat)==difflat)[0][0]

    # create offset to center around area
    # Npixx       =   Nx
    # Npixy       =   Ny
    ix          =   int((ilon+1) - np.floor(Nx/2))
    iy          =   int((ilat+1) - np.floor(Ny/2))

    # read Data
    band        =   ds.GetRasterBand(1)

    LC          =   band.ReadAsArray(ix,iy,Nx,Ny)
    lon_sub     =   lon[ix]
    lat_sub     =   lat[iy]
    return LC, lon_sub, lat_sub


    # modify ...
def Read_GLAS(lon_s=05.2913,lat_s = 52.1326, Nx=1, Ny=1):
    import gdal



    file_glas   = '/home/ucfajti/Data/Satellites/GLAS/2005/Simard_Pinto_3DGlobalVeg_L3C.tif'

    # open file
    ds          =   gdal.Open(file_glas)

    # determine coordinates
    Nlon        =   ds.RasterXSize
    Nlat        =   ds.RasterYSize
    gt          =   ds.GetGeoTransform()

    minlon      =   gt[0]
    maxlon      =   gt[0] + Nlon*gt[1] + Nlat*gt[2]
    minlat      =   gt[3]
    maxlat      =   gt[3] + Nlon*gt[4] + Nlat*gt[5]

    lon         =   np.linspace(minlon,maxlon,Nlon)
    lat         =   np.linspace(minlat,maxlat,Nlat)

    # Selecct pixel(s)
    difflon     =   np.abs(lon-lon_s)
    difflat     =   np.abs(lat-lat_s)

    ilon        =   np.where(min(difflon)==difflon)[0][0]
    ilat        =   np.where(min(difflat)==difflat)[0][0]

    # create offset to center around area
    # Npixx       =   Nx
    # Npixy       =   Ny
    ix          =   int((ilon+1) - np.floor(Nx/2))
    iy          =   int((ilat+1) - np.floor(Ny/2))

    # read Data
    band        =   ds.GetRasterBand(1)

    hc_forest   =   band.ReadAsArray(ix,iy,Nx,Ny)
    lon_sub     =   lon[ix]
    lat_sub     =   lat[iy]
    return hc_forest, lon_sub, lat_sub




    return Veg_Class_new

def translate_IGBP2NLDAS(Veg_Class):
    Class                                   =   dict()
    Class['water']                          =   [ 0,  0]
    Class['evergreen_needleleaf_forest']    =   [ 1,  1]
    Class['evergreen_broadleaf_forest']     =   [ 2,  2]
    Class['deciduous_needleleaf_forest']    =   [ 3,  3]
    Class['deciduous_broadleaf_forest']     =   [ 4,  4]
    Class['Mixed_forest']                   =   [ 5,  5]
    Class['closed_shrublands']              =   [ 6,  8]
    Class['open_shrublands']                =   [ 7,  9] #?
    Class['Woody_savannas']                 =   [ 8,  6] #?
    Class['Savannas']                       =   [ 9,  5]
    Class['grassland']                      =   [10, 10]
    Class['permanent_wetlands']             =   [11,  0]
    Class['croplands']                      =   [12, 11]
    Class['uran_and_builtup']               =   [13, 13]
    Class['mixed_cropland_natural']         =   [14, 11]
    Class['snow_ice']                       =   [15,  0]
    Class['barren']                         =   [16, 12]
    Class['unclassified']                   =   [17,  0]
    Class['fill']                           =   [18,  0]

    Veg_Class_new                           =   Veg_Class + 0
    for class_v in Class.itervalues():
        class_v_FAO                         =   class_v[0]
        class_v_nldas                       =   class_v[1]
        iClass                              =   class_v_FAO==Veg_Class
        Veg_Class_new[iClass]               =   class_v_nldas


    return Veg_Class_new
def translate_FAO2NLDAS(Veg_Class):
    Class                                           =   dict()
    Class['irrigated croplands']                    =   [ 11, 11] # Post-flooding or irrigated croplands (or aquatic)
    Class['Rainfed croplands']                      =   [ 14, 11] # Rainfed croplands
    Class['Mosaic cropland vegetation']             =   [ 20, 11] # Mosaic cropland (50-70%) / vegetation (grassland/shrubland/forest) (20-50%)
    Class['Mosaic vegetation cropland']             =   [ 30, 11] # Mosaic vegetation (grassland/shrubland/forest) (50-70%) / cropland (20-50%)
    Class['Closed Open broadleaved forest']         =   [ 40,  2] # Closed to open (>15%) broadleaved evergreen or semi-deciduous forest (>5m)
    Class['Closed broadleaved deciduous forest']    =   [ 50,  4] # Closed (>40%) broadleaved deciduous forest (>5m)
    Class['woodland']                               =   [ 60,  6] # Open (15-40%) broadleaved deciduous forest/woodland (>5m)
    Class['Closed needleleaved evergreen forest']   =   [ 70,  1] # Closed (>40%) needleleaved evergreen forest (>5m)
    Class['Open needleleaved evergreen forest']     =   [ 90,  3] # Open (15-40%) needleleaved deciduous or evergreen forest (>5m)
    Class['Closed Open Mixed forest']               =   [100,  1] # Closed to open (>15%) mixed broadleaved and needleleaved forest (>5m)
    Class['Mosaic forest grassland']                =   [110,  5] # Mosaic forest or shrubland (50-70%) / grassland (20-50%)
    Class['Mosaic grassland forest']                =   [120,  7] # Mosaic grassland (50-70%) / forest or shrubland (20-50%)
    Class['shrubland']                              =   [130,  9]# Closed to open (>15%) (broadleaved or needleleaved, evergreen or deciduous) shrubland (<5m)
    Class['grassland']                              =   [140, 10] # Closed to open (>15%) herbaceous vegetation (grassland, savannas or lichens/mosses)
    Class['Sparse']                                 =   [150, 11] # Sparse (<15%) vegetation
    Class['flooded  broadleaved forest']            =   [160,  2] # Closed to open (>15%) broadleaved forest regularly flooded (semi-permanently or temporarily) - Fresh or brackish water
    Class['flooded shrubland']                      =   [170,  9] # (>40%) broadleaved forest or shrubland permanently flooded - Saline or brackish water
    Class['flooded grassland']                      =   [180, 10] # Closed to open (>15%) grassland or woody vegetation on regularly flooded or waterlogged soil - Fresh, brackish or saline water
    Class['Urban']                                  =   [190, 13] # Artificial surfaces and associated areas (Urban areas >50%)
    Class['Bare areas']                             =   [200, 12] # Bare areas
    Class['Water bodies']                           =   [210,  0] # Water bodies
    Class['Permanent snow and ice']                 =   [220,  0] # Permanent snow and ice
    Class['No data']                                =   [230,  0] # No data (burnt areas, clouds,?)


    Veg_Class_new                           =   Veg_Class + 0
    for class_v in Class.itervalues():
        class_v_FAO                         =   class_v[0]
        class_v_nldas                       =   class_v[1]
        iClass                              =   class_v_FAO==Veg_Class
        Veg_Class_new[iClass]               =   class_v_nldas


    return Veg_Class_new

def determine_vegetation_height_parameters():
    # Parameters
    ## Define land surface parameters for different NLDAS classes
    # Min/Max Vegetation Height (database) http://ldas.gsfc.nasa.gov/nldas/web/web.veg.table.html

    LC          = np.zeros([14,1])
    LC[ 0,:]    =   [ 0]    #Water / Goode's Interrupted Space
    LC[ 1,:]    =   [ 1]  #Evergreen Needleleaf Forest
    LC[ 2,:]    =   [ 2]  #Evergreen Broadleaf Forest
    LC[ 3,:]    =   [ 3]  #Deciduous Needleleaf Forest
    LC[ 4,:]    =   [ 4]  #Deciduous Broadleaf Forest
    LC[ 5,:]    =   [ 5]  #Mixed Cover
    LC[ 6,:]    =   [ 6]  #Woodland
    LC[ 7,:]    =   [ 7]  #Wooded Grassland
    LC[ 8,:]    =   [ 8]  #Closed Shrubland
    LC[ 9,:]    =   [ 9]  #Open Shrubland
    LC[10,:]    =   [10]  #Grassland
    LC[11,:]    =   [11]  #Cropland
    LC[12,:]    =   [12]  #Bare Ground
    LC[13,:]    =   [13]  #Urban and Built-Up

    # vegetation structural parameters. Please note that the first entry is the modelled version and the 2nd entry is the measured version
    hc_max          =   np.zeros([14,2])
    hc_max[ 0,:]    =   [ 0.00,     0.00]    #Water / Goode's Interrupted Space
    hc_max[ 1,:]    =   [17.00, 	17.00]  #Evergreen Needleleaf Forest
    hc_max[ 2,:]    =   [35.00, 	35.00]  #Evergreen Broadleaf Forest
    hc_max[ 3,:]    =   [15.50, 	14.00]  #Deciduous Needleleaf Forest
    hc_max[ 4,:]    =   [20.00, 	20.00]  #Deciduous Broadleaf Forest
    hc_max[ 5,:]    =   [19.25, 	 8.00]  #Mixed Cover
    hc_max[ 6,:]    =   [14.30, 	14.11]  #Woodland
    hc_max[ 7,:]    =   [ 7.04, 	 8.31]  #Wooded Grassland
    hc_max[ 8,:]    =   [ 0.60, 	 5.27]  #Closed Shrubland
    hc_max[ 9,:]    =   [ 0.51, 	 4.14]  #Open Shrubland
    hc_max[10,:]    =   [ 0.56, 	 0.60]  #Grassland
    hc_max[11,:]    =   [ 0.55, 	 0.60]  #Cropland
    hc_max[12,:]    =   [ 0.20, 	 0.00]  #Bare Ground
    hc_max[13,:]    =   [ 6.02, 	 0.00]  #Urban and Built-Up

    hc_min          = np.zeros([14,2])
    hc_min[ 0,:]    =   [ 0.00,     0.00]    #Water / Goode's Interrupted Space 	N/A 	NONE 	---
    hc_min[ 1,:]    =   [ 8.50, 	 6.00]  #Evergreen Needleleaf Forest
    hc_min[ 2,:]    =   [ 1.00, 	 1.00]  #Evergreen Broadleaf Forest 	1 	1
    hc_min[ 3,:]    =   [ 7.50, 	 2.00]  #Deciduous Needleleaf Forest
    hc_min[ 4,:]    =   [11.50, 	11.00]  #Deciduous Broadleaf Forest
    hc_min[ 5,:]    =   [10.00, 	 4.00]  #Mixed Cover
    hc_min[ 6,:]    =   [ 7.52, 	 5.55]  #Woodland
    hc_min[ 7,:]    =   [ 3.60, 	 3.68]  #Wooded Grassland
    hc_min[ 8,:]    =   [ 0.74, 	 1.87]  #Closed Shrubland
    hc_min[ 9,:]    =   [ 0.08, 	 0.00]  #Open Shrubland
    hc_min[10,:]    =   [ 0.04, 	 0.00]  #Grassland
    hc_min[11,:]    =   [ 0.05, 	 0.00]  #Cropland
    hc_min[12,:]    =   [ 0.05, 	 0.00]  #Bare Ground
    hc_min[13,:]    =   [ 3.59, 	 0.00]  #Urban and Built-Up

    # The values need to be determined by apriori investigation into the NDVI timeseries for each LC type!
    NDVI_max          = np.zeros([14,2])
    NDVI_max[ 0,:]    =   [ 0.00,     0.00]    #Water / Goode's Interrupted Space 	N/A 	NONE 	---
    NDVI_max[ 1,:]    =   [17.00, 	17.00]  #Evergreen Needleleaf Forest
    NDVI_max[ 2,:]    =   [35.00, 	35.00]  #Evergreen Broadleaf Forest
    NDVI_max[ 3,:]    =   [15.50, 	14.00]  #Deciduous Needleleaf Forest
    NDVI_max[ 4,:]    =   [20.00, 	20.00]  #Deciduous Broadleaf Forest
    NDVI_max[ 5,:]    =   [19.25, 	 8.00]  #Mixed Cover
    NDVI_max[ 6,:]    =   [14.30, 	14.11]  #Woodland
    NDVI_max[ 7,:]    =   [ 7.04, 	 8.31]  #Wooded Grassland
    NDVI_max[ 8,:]    =   [ 0.60, 	 5.27]  #Closed Shrubland
    NDVI_max[ 9,:]    =   [ 0.51, 	 4.14]  #Open Shrubland
    NDVI_max[10,:]    =   [ 0.56, 	 0.60]  #Grassland
    NDVI_max[11,:]    =   [ 0.55, 	 0.60]  #Cropland
    NDVI_max[12,:]    =   [ 0.20, 	 0.00]  #Bare Ground
    NDVI_max[13,:]    =   [ 6.02, 	 0.00]  #Urban and Built-Up
    NDVI_max[ :,:]    =   [ 0.80, 	 0.80]  #temporary solution according to wang et al 2008

    NDVI_min          = np.zeros([14,2])
    NDVI_min[ 0,:]    =   [ 0.00,     0.00]    #Water / Goode's Interrupted Space 	N/A 	NONE 	---
    NDVI_min[ 1,:]    =   [ 8.50, 	 6.00]  #Evergreen Needleleaf Forest
    NDVI_min[ 2,:]    =   [ 1.00, 	 1.00]  #Evergreen Broadleaf Forest 	1 	1
    NDVI_min[ 3,:]    =   [ 7.50, 	 2.00]  #Deciduous Needleleaf Forest
    NDVI_min[ 4,:]    =   [11.50, 	11.00]  #Deciduous Broadleaf Forest
    NDVI_min[ 5,:]    =   [10.00, 	 4.00]  #Mixed Cover
    NDVI_min[ 6,:]    =   [ 7.52, 	 5.55]  #Woodland
    NDVI_min[ 7,:]    =   [ 3.60, 	 3.68]  #Wooded Grassland
    NDVI_min[ 8,:]    =   [ 0.74, 	 1.87]  #Closed Shrubland
    NDVI_min[ 9,:]    =   [ 0.08, 	 0.00]  #Open Shrubland
    NDVI_min[10,:]    =   [ 0.04, 	 0.00]  #Grassland
    NDVI_min[11,:]    =   [ 0.05, 	 0.00]  #Cropland
    NDVI_min[12,:]    =   [ 0.05, 	 0.00]  #Bare Ground
    NDVI_min[13,:]    =   [ 3.59, 	 0.00]  #Urban and Built-Up
    NDVI_min[ :,:]    =   [ 0.01, 	 0.01]  #temporary solution according to wang et al 2008

    leafwidth       = np.zeros([14,2])
    leafwidth[ 0,:] =   [0.000, 	0.000] #Water / Goode's Interrupted Space
    leafwidth[ 1,:] =   [0.001, 	0.001] #Evergreen Needleleaf Forest
    leafwidth[ 2,:] =   [0.050, 	0.070] #Evergreen Broadleaf Forest
    leafwidth[ 3,:] =   [0.001, 	0.001] #Deciduous Needleleaf Forest
    leafwidth[ 4,:] =   [0.080, 	0.080] #Deciduous Broadleaf Forest
    leafwidth[ 5,:] =   [0.040, 	0.040] #Mixed Cover
    leafwidth[ 6,:] =   [0.019, 	0.022] #Woodland
    leafwidth[ 7,:] =   [0.018, 	0.023] #Wooded Grassland
    leafwidth[ 8,:] =   [0.006, 	0.015] #Closed Shrubland
    leafwidth[ 9,:] =   [0.003, 	0.013] #Open Shrubland
    leafwidth[10,:] =   [0.010, 	0.010] ##Grassland
    leafwidth[11,:] =   [0.010, 	0.010] #Cropland
    leafwidth[12,:] =   [0.003, 	0.000] #Bare Ground
    leafwidth[13,:] =   [0.015, 	0.000] #Urban and Built-Up

    # veg_traits_      =   zip(LC,hc_min[:,0],hc_max[:,0], ndvi_max[:,0], ndvi_min[:,0],leafwidth[:,0])
    # return veg_traits_
    return LC, hc_max[:,0], hc_min[:,0], NDVI_max[:,0], NDVI_min[:,0], leafwidth[:,0]
def determine_vegetation_height(LC_Values=11, NDVI=0.2, Hc_tall=20):

    lc_, hc_max_, hc_min_, ndvi_max_, ndvi_min_, leafwidth_     =   determine_vegetation_height_parameters()

    Hc_min                                                      =   LC_Values*0.
    Hc_max                                                      =   LC_Values*0.
    NDVI_min                                                    =   LC_Values*0.
    NDVI_max                                                    =   LC_Values*0.
    Leafwidth                                                   =   LC_Values*0.

    # print lc_
    for lc,hc_min,hc_max,ndvi_max, ndvi_min, leafwidth in zip(lc_, hc_max_, hc_min_, ndvi_max_, ndvi_min_, leafwidth_):
        # print lc,hc_min,hc_max
        itype                                                   =   lc==LC_Values

        # print LC_Values,itype
        Hc_min[itype]                                           =   hc_min
        Hc_max[itype]                                           =   hc_max
        NDVI_min[itype]                                         =   ndvi_min
        NDVI_max[itype]                                         =   ndvi_max
        Leafwidth[itype]                                        =   leafwidth
    # make sure that NDVI does not exceed range of NDVImin and NDVImax


    Hc_low                                                      =   (Hc_min+ (Hc_max - Hc_min)*(NDVI-NDVI_max)/(NDVI_min-NDVI_max + 1e-2))
    # Hc_tall, Hc_lon, Hc_lat                                     =   Read_GLAS(lon_s=lon_s,lat_s= lat_s, Nx=Nx, Ny=Ny)

    # resampe to correct for resolution differences and stuff like that

    #
    # combine heights
    ierror                                                      =   Hc_tall==0
    Hc                                                          =   Hc_tall*1.
    Hc[ierror]                                                  =   Hc_low[ierror]

    plt.ion()
    plt.imshow(Hc)

    return Hc, Leafwidth

class SCOPE ( ):
    """A standard state configuration for the CMEM model required for eoldas"""
    def __init__ ( self,options =   '-nosplash', scoperoot = '~/Simulations/Matlab/SCOPE/SCOPE_v1.xx/code/'):
        import matlab_wrapper
        # matlab                                 = matlab_wrapper.MatlabSession(options=options,matlab_root='/usr/local/matlab_r2014a/')

        import socket
        hostname                                                =   socket.gethostname()
        if hostname=='hephaestus.geog.ucl.ac.uk':
            matlab_root='/usr/local/matlab_r2014a/'
        else:
            matlab_root='/usr/local/matlab_r2016a/'


        matlab                                                  =   matlab_wrapper.MatlabSession(options=options,matlab_root=matlab_root)


        # options=options, buffer_size=buffer_size
        ######################################################
        ############# Initialize SCOPE model #################
        ######################################################

        # change matlab working directory to SCOPE directory
        matlab.put('scoperoot',scoperoot)
        matlab.eval('cd(scoperoot)')

        # Import default values to run scope
        matlab.eval('SCOPE')

        # Translate the Parameter input-structures to arrays in matlab workspace
        matlab.eval('SCOPE_translate_fwd')

        # Retrieve values Input Parameters from initial set
        Names                                                   =   ['Ta', 'p', 'ea', 'u', 'Rin', 'Rli', 'Cab', 'Cca', 'Cdm', 'Cw', 'Cs', 'N' , 'emissivity_leaf', 'emissivity_soil', 'Vcmo', 'lai', 'hc', 'lw', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'tts','tto','psi']



        Input_real                                              =   OrderedDict()
        for varname in Names:
            Input_real[varname]                                 =   matlab.get(varname)

        # input_parameters                                        =   ['Ta', 'p',           'ea',  'u', 'Rin',               'Rli', 'Cab', 'Cca','Cdm', 'Cw', 'Cs', 'N', 'emissivity_leaf', 'emissivity_soil', 'Vcmo', 'lai', 'hc', 'lw']
        # default_values                                          =   [22.05,993.8/1.0e3,  67.8,  2.3,  440./1.0e3,  381.69/1.0e3,  0.89,  0.99, 0.38, 0.78, 0.60, 1.6,         0.98,                    0.95,    30., 0.95,  49.55,0.054] #transformed

        self.matlab                                             =   matlab
        self.Input_real                                         =   Input_real
        self.Input_transformed                                  =   self.forward_transform(copy.deepcopy(Input_real))
        self.wl                                                 =   matlab.get('wlS')
        self.names                                              =   Names

    def update_input(self,Input_transformed_new):
        Input_transformed                                       =   self.Input_transformed
        matlab = self.matlab

        # update Python workspace
        for varname,varvalue in Input_transformed_new.iteritems():
            if varname in [name for name in Input_transformed.iterkeys()]:
                Input_transformed[varname]                      =   Input_transformed_new[varname]

        # update default
        self.Input_transformed                                  =   Input_transformed

        # transfer values to list
        xx                                                      =   [v for v in Input_transformed.itervalues()]
        # xx                                                      =   xx[0:22]
        tts                                                     =   Input_transformed['tts']
        tto                                                     =   Input_transformed['tto']
        psi                                                     =   Input_transformed['psi']

        return xx, tts, tto, psi


    def run_model(self, xx_transformed, sza, vza, raa ):
        Input_transformed                                       =   self.Input_transformed
        Names                                                   =   [name for name in Input_transformed.iterkeys()]

        Input_transformed_new                                   =   OrderedDict()
        for i,name in enumerate(xx_transformed):
            Input_transformed_new[Names[i]]                     =   xx_transformed[i]

        Input_transformed_new['tts']                            =   sza
        Input_transformed_new['tto']                            =   vza
        Input_transformed_new['psi']                            =   raa

        Output                                                  =   self.run_model_with_dict(Input_transformed_new)

        # matlab = self.matlab
        #
        # for x in xx
        # Input                                                   =   self.Input
        # ######################################################
        # ############# Run SCOPE model ########################
        # ######################################################
        # # update Python workspace
        # for varname,varvalue in Input_new.iteritems():
        #     if varname in [name for name in Input.iterkeys()]:
        #         Input[varname]                                  = Input_new[varname]
        #
        # # Update parameter arrays in matlab workspace
        # for varname,varvalue in Input.iteritems():
        #     matlab.put(varname,varvalue)
        #
        # # Translate the Parameter arrays to input-structures in matlab workspace
        # matlab.eval('SCOPE_translate_bck')
        #
        # # Run scope with updates workspace
        # matlab.eval('SCOPE_run')

        return Output

    def run_model_with_dict(self,Input_transformed_new):
        input_transformed                                       =   self.Input_transformed
        matlab                                                  =   self.matlab
        inverse_transform                                       =   self.inverse_transform
        retrieve_output                                         =   self.retrieve_output

        ######################################################
        ############# Pre processing  ########################
        ######################################################


        # # redefine z, to make sure that measurement height is above canopy height!
        # hc_meas_above_can           =   Input_transformed_new['z']
        # hc_can                      =   Input_transformed_new['hc']
        # hc_meas                     =   hc_meas_above_can+ hc_can
        # Input_transformed_new['z']  =   hc_meas

        ######################################################
        ############# Run SCOPE model ########################
        ######################################################

        # update Python workspace
        for varname,varvalue in Input_transformed_new.iteritems():
            if varname in [name for name in input_transformed.iterkeys()]:
                input_transformed[varname]                      =   Input_transformed_new[varname]

        # apply inverse transformation
        input_real                                              =   inverse_transform (input_transformed )

        # update matlab workspace
        for varname,varvalue in input_real.iteritems():
            matlab.put(varname,varvalue)
            # print varname + '\t%f' % varvalue
        # matlab.eval('clear canopy spectral meteo')
        matlab.eval('SCOPE_translate_bck')


        # Run scope with updates workspace
        matlab.eval('SCOPE_run')


        # Run scope with updates workspace
        Output                                                  =   retrieve_output()

        return Output

    def inverse_transform (self, Input ):
        """Inverse transform the SCOPE and PROSAIL parameters"""
        for varname,varvalue in Input.iteritems():
            if varname == 'Cab':
                varvalue = -100.*np.log ( varvalue )
            elif varname == 'Cca':
                varvalue = -100.*np.log ( varvalue )
            elif varname == 'Cw':
                varvalue = (-1./50.)*np.log ( varvalue )
            elif varname == 'Cdm':
                varvalue = (-1./100.)*np.log ( varvalue )
            elif varname == 'lai':
                varvalue = -2.*np.log ( varvalue )
            elif varname=='ALA':
                varvalue = 90.*varvalue
            elif varname=='p':
                varvalue    =   varvalue*1.0e3
            elif varname=='Rin':
                varvalue    =   varvalue*1.0e3
            elif varname=='Rli':
                varvalue    =   varvalue*1.0e3
            Input[varname]  =   varvalue*1.

            # self.Input  = Input
        return Input

    def forward_transform (self, Input ):
        for varname,varvalue in Input.iteritems():
            if varname == 'Cab':
                varvalue    =   np.exp((1/-100.*varvalue))
            elif varname == 'Cca':
                varvalue    =   np.exp((1/-100.*varvalue))
            elif varname == 'Cw':
                varvalue    =   np.exp((-50.*varvalue))
            elif varname == 'Cdm':
                varvalue    =   np.exp((-100.*varvalue))
            elif varname == 'lai':
                varvalue    =   np.exp((-1/2.*varvalue))
            elif varname=='ALA':
                varvalue    =   varvalue/90.
            elif varname=='p':
                varvalue    =   varvalue/1.0e3
            elif varname=='Rin':
                varvalue    =   varvalue/1.0e3
            elif varname=='Rli':
                varvalue    =   varvalue/1.0e3


            Input[varname]  =   varvalue*1.

            # self.Input  = Input
        return Input

    def retrieve_output(self):
        matlab = self.matlab

        # update matlab workspace
        matlab.eval('SCOPE_post1')


        # rsd     =   rad.rsd;                            # [nwl x 1 double]
        # rdd     =   rad.rdd;                            # [nwl x 1 double]
        # rdo     =   rad.rdo;                            # [nwl x 1 double]
        # rso     =   rad.rso;                            # [nwl x 1 double]

        # Lo_     =   rad.Lo_;                            # [nwl x 1 double]      %directional radiance
        # 'rsd','rdd','rdo',
        Names       =   ['rso','Lo_','Lot_','Ltot_','Ts','Tcu','Tch','Rntot','lEtot','Htot','Gtot', 'Bandnr','Bandwl','Tb']


        Output = OrderedDict()
        for varname in Names:
            Output[varname]                                     =   matlab.get(varname)

        return Output


    def plot_results(self):
        matlab                                                  =   self.matlab

        matlab.eval('SCOPE_plot')


if __name__ == "__main__":
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    from eoldas_ng import *
    import copy
    # from datetime import datetime as datet
    import time
    import datetime
    import gdal
    ###########################################################################################
    # The following shows how SCOPE can be run from Python using different methods (using lists or dictionaries)'
    ###########################################################################################

    print 'Start SCOPE wrapper'

    # define parameters
    print '- Define Parameters'
    lon_s                                                       =   05.2913
    lat_s                                                       =   52.1326
    Nx                                                          =   1
    Ny                                                          =   1


    print '- initialize SCOPE'
    tic                                                         =   dt.datetime.now()
    # scope_model                                                 =   SCOPE(options='-nojvm -nosplash')
    scope_model                                                 =   SCOPE(options='-nosplash')
    do_forward_model                                            =   scope_model.run_model
    Input_real                                                  =   copy.deepcopy(scope_model.Input_real)
    toc                                                         =   dt.datetime.now()

    ###########################################################################################
    # test reading data
    ###########################################################################################
    # Read vegetation height (please note that NDVImin/NDVImax need to be determined for each pixel using multiple years of NDVI)
    print '- Read Data'
    print '\t * NDVI'
    NDVI                                                        =   0.2
    print '\t * LandCover Type (GlobCover)'
    LC_FAO, LC_lon_, LC_lat_                                    =   Read_GlobCover(lon_s=lon_s,lat_s = lat_s, Nx=Nx, Ny=Ny)
    print '\t * height of canopy (forest)'
    Hc_tall, Hc_lon, Hc_lat                                     =   Read_GLAS(lon_s=lon_s,lat_s= lat_s, Nx=Nx, Ny=Ny)

    # resampling data
    if Nx<>1 or Ny<>1:
        print 'this needs to be sorted'
        # gdal.ReprojectImage

    # Preprocessing
    print '- PreProcessing'
    LC_Values                                                   =   translate_FAO2NLDAS(LC_FAO)
    Hc, leafwidth                                               =   determine_vegetation_height(LC_Values=LC_Values, NDVI=0.2, Hc_tall=Hc_tall) # determine_vegetation_height(lon_s=lon_s,lat_s = lat_s , Nx=Nx, Ny=Ny, NDVI=NDVI)


    ###########################################################################################
    # test forward and inverse transformation
    ###########################################################################################
    # print Input
    # Input_transformed = scope_model.forward_transform(Input)
    # print Input_transformed
    #
    # Input_regular = scope_model.inverse_transform(Input_transformed)
    # print Input_regular

    ###########################################################################################
    # test running SCOPE using two methods
    ###########################################################################################

    plt.close()

    print '- run SCOPE (dict method)'
    Input_real_new                                              =   copy.deepcopy(Input_real)
    Input_real_new['Ta']                                        =   25
    Input_real_new['Vcmo']                                      =   30
    Input_transformed_new                                       =   scope_model.forward_transform(Input_real_new)
    Output1                                                      =   scope_model.run_model_with_dict(Input_transformed_new)
    plt.plot(Output1['Tb'])

    Input_real_new                                              =   copy.deepcopy(Input_real)
    Input_real_new['Vcmo']                                      =   30
    Input_transformed_new                                       =   scope_model.forward_transform(Input_real_new)
    Output2                                                      =   scope_model.run_model_with_dict(Input_transformed_new)
    plt.plot(Output2['Tb'],'--')
    plt.legend(['Vcmo = 20', 'Vcmo = 30'])
    diffO =  Output1['Tb']-Output2['Tb']
    print diffO[5:7]


    scope_model                                                 =   SCOPE(options='-nosplash')
    print '- run SCOPE (list method )'
    Input_real_new                                              =   copy.deepcopy(scope_model.Input_real)
    Input_real_new['Ta']                                        =   26
    Input_real_new['Vcmo']                                      =   33.3332
    Input_transformed_new                                       =   scope_model.forward_transform(Input_real_new)
    xx_transformed,tts,tto,psi                                  =   scope_model.update_input(Input_transformed_new)
    Output                                                      =   scope_model.run_model(xx_transformed,tts,tto,psi)
    plt.plot(Output['Tb'],'--')

    Input_real_new                                              =   copy.deepcopy(scope_model.Input_real)
    Input_real_new['Ta']                                        =   26
    Input_real_new['Vcmo']                                      =   50.
    Input_transformed_new                                       =   scope_model.forward_transform(Input_real_new)
    xx_transformed,tts,tto,psi                                  =   scope_model.update_input(Input_transformed_new)
    Output                                                      =   scope_model.run_model(xx_transformed,tts,tto,psi)
    plt.plot(Output['Tb'],'--')

    ###########################################################################################
    # test running SCOPE consecutely
    ###########################################################################################
    print '- Timings of SCOPE'
    N                                                           =   10
    Input_new                                                   =   copy.deepcopy(Input_real)
    Input_transformed_new                                       =   scope_model.forward_transform(Input_real_new)
    print 'Start SCOPE wrapper'

    # define parameters
    print '- Define Parameters'
    lon_s                                                       =   05.2913
    lat_s                                                       =   52.1326
    Nx                                                          =   1
    Ny                                                          =   1


    print '- initialize SCOPE'
    tic                                                         =   dt.datetime.now()
    # scope_model                                                 =   SCOPE(options='-nojvm -nosplash')
    scope_model                                                 =   SCOPE(options='-nosplash')
    do_forward_model                                            =   scope_model.run_model
    Input_real                                                  =   copy.deepcopy(scope_model.Input_real)
    toc                                                         =   dt.datetime.now()

    tic                                                         =   dt.datetime.now()
    for i in xrange(0,N):
        Output                                                  =   scope_model.run_model_with_dict(Input_transformed_new)
    toc                                                         =   dt.datetime.now()
    print '\t * Dict Method lasts %f seconds for N=%1.0f runs' % ((toc - tic).total_seconds(),N)

    tic                                                         =   dt.datetime.now()
    for i in xrange(0,N):
        xx_transformed,tts,tto,psi                              =   scope_model.update_input(Input_transformed_new)
        Output                                                  =   scope_model.run_model(xx_transformed,tts,tto,psi)
    toc                                                         =   dt.datetime.now()
    print '\t * List Method lasts %f seconds for N=%1.0f runs' % ((toc - tic).total_seconds(),N)

    ###########################################################################################
    # test storing output consecutively
    ###########################################################################################
    Output                                                      =   []
    xx_transformed,tts,tto,psi                                  =   scope_model.update_input(Input_transformed_new)
    Output.append(scope_model.run_model(xx_transformed,tts,tto,psi))
    Output.append(scope_model.run_model(xx_transformed,tts,tto,psi))


    # print Output
    # print '- plot SCOPE'
    # scope_model.plot_results()


    ###########################################################################################
    # test running SCOPE consecutely
    ###########################################################################################

    # print ' - transfer data to Python workspace'
    # Output                                                      =   scope_model.retrieve_output()
