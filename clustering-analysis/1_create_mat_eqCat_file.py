#------------------------------------------------------------------------------
import os

#------------------------------my modules-------------------------------------- 
from src.EqCat import EqCat
eqCat = EqCat( )

#=================================1==============================================
#                            dir, file, params
#================================================================================

dir_in = 'data'
file_in = 'Tohoku_eqs.txt'

#=================================2==============================================
#                            load data
#================================================================================
import numpy as np
# 0-5 (datetime), 6(ID), 7 (lat), 8 (lon), 9 (depth), 10 (mag)
#mData = np.loadtxt( f"{dir_in}/{file_in}", delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6,7,8,9,10)).T
#print( mData.shape)

eqCat.loadEqCat( f"{dir_in}/{file_in}", 'USGS', usecols = (0,11,1,2,3,4))

print( 'total no. of events: ', eqCat.size())
print( sorted( eqCat.data.keys()))
#=================================3==============================================
#                     test plot and save to .mat binary
#================================================================================
eqCat.saveMatBin( file_in.replace( 'txt', 'mat'))
newEqCat = EqCat( )
newEqCat.loadMatBin( file_in.replace( 'txt', 'mat'))
print( newEqCat.size())
print( sorted( newEqCat.data.keys()))







