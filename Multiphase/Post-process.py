#!/usr/bin/env python3

#-- Import general libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=18)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

# # # ######################################### # # #
# # # ############ Post processing ############ # # #
# # # ######################################### # # #
path_data = 'Results/Water_Column_3D_256'
path_figure = 'Figures/Water_Column_3D_256'
checkpoint = 50
for i in range(1,checkpoint+1):
    rho = np.load(path_data+'/rho'+str(i*400)+'.npy').astype('float32')   

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(231)
    plt.imshow((rho)[:,:,10], cmap='RdBu',vmin=200,vmax=1000)
    # plt.colorbar()
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title('West surface',fontsize=15)
    ax1 = plt.subplot(232)
    plt.imshow((rho)[:,10,:], cmap='RdBu',vmin=200,vmax=1000)
    # plt.colorbar()
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title('South surface',fontsize=15)
    ax1 = plt.subplot(233)
    plt.imshow((rho)[10,:,:], cmap='RdBu',vmin=200,vmax=1000)
    # plt.colorbar()
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title('Bottom surface',fontsize=15)
    ax1 = plt.subplot(234)
    plt.imshow((rho)[:,:,245], cmap='RdBu',vmin=200,vmax=1000)
    # plt.colorbar()
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title('East surface',fontsize=15)
    ax1 = plt.subplot(235)
    plt.imshow((rho)[:,245,:], cmap='RdBu',vmin=200,vmax=1000)
    # plt.colorbar()
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title('North surface',fontsize=15)
    ax1 = plt.subplot(236)
    plt.imshow((rho)[245,:,:], cmap='RdBu',vmin=200,vmax=1000)
    # plt.colorbar()
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.title('Top surface',fontsize=15) 
    
    if i < 10:
        save_name = path_figure+"/0"+str(i)+".jpg"
    else:
        save_name = path_figure+"/"+str(i)+".jpg"
#     plt.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.savefig(save_name)
    plt.close()

    print('Job done!')
