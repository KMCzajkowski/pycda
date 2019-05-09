from pyCDA import * ### JUST IMPORT THE WHOLE THING, imports also numpy, matplotlib, h5py and os
import sys # sys import is required for 
import matplotlib.pyplot as plt
################################## USER INPUT #################################################

if __name__ == '__main__':
    # here is the main calculation
    # Lets define some parameters for the simulation
    start_wave =250.0E-9
    end_wave = 500.0E-9
    wave_spacing = 4.0E-9
    solver_tolerance = 1E-6 # NOT NECESSARY - WILL BE ELIMINATED SOMETIME IN THE FUTURE

    # Prepare the wavelength vector
    wave = np.arange(start_wave, end_wave, wave_spacing)
    
    # Prepare materials
    eps=set_drude(wave,9.1492,0.133,3) #Drude material with parameters: omega_res (eV), gamma (eV), eps_inf
    eps2=load_nk(wave,'files/PdJohnson.csv') 
    #eps=load_nk(wave,'files/PdJohnson.csv')
    epsvec=np.vstack((eps,eps2)) #input must be in that form
    
    #eps = LD(wave, material='Ag', model='LD')  # Creates silver object with dielectric function of LD model - experimental feature
    n_surround = 1 # Refractive index of the surrounding

    #definition of the incident field
    E0_vec = np.array([0, 0, 1], dtype=complex)  # incident polarization of the light, [1,0,0] means light is polarized in x, [0,1,0] mean light is polarized in y
    k_vec = np.array([1, 0, 0]) # incident k vector [0,0,1] means light is travelling in +z axis

    eps_surround = n_surround ** 2 # Dielectric function of surrounding
    
    # EXAMPLES OF POSITION GENERATION
    # Manual position generation
    sysarg1=sys.argv[1]
    sysarg2=sys.argv[2]
    sysarg3=sys.argv[3]
    seria="PdAgDimerR"+sysarg1.replace(".","_")+"th"+sysarg2.replace(".","_")+"phi"+sysarg3.replace(".","_") # Name for hdf5 file 
    N=2
    phi=pi*float(sys.argv[3]) # third terminal parameter
    th=pi*float(sys.argv[2]) # second terminal parameter
    delta=float(sys.argv[1])*1e-9 #first terminal parameter
    r_eff=np.array([30e-9,3e-9])
    R=r_eff[0]+r_eff[1]+delta
    pos=np.array([[0,0,0],[R*np.sin(th)*np.cos(phi),R*np.sin(th)*np.sin(phi),R*np.cos(th)]])
    
    # Load particles from matlab
    #N, pos, r_eff =loadmlabpos('kc0.33333_D60_N1750_39',2.5)

    # Create particles of identical radius on grid
    #N, pos, r_eff = create_particle_on_grid(nx=2, ny=1, step_x=2.5*200*30E-9, step_y=2.5*2*30E-9,radius=30E-9) #all of these inputs have some default values

    # Single particle
    #N, pos, r_eff = create_particle_on_grid(radius=30E-9)


    # Create particles of random diameter and position - experimental feature
    #N, pos, r_eff = random_particles_not_touching(N=200, x_span=100E-9, y_span=100E-9,z_span=40E-9, max_radius = 10E-9, min_radius = 2E-9)

################################### CALCULATION PART ############################################################

    q_ext,q_abs,q_scat,p_calc,c_extind,c_absind,field_enh=cdacalc(wave,epsvec,eps_surround,N,pos,r_eff,E0_vec,k_vec)

################################### OUTPUT GENERATION ###########################################################

    np.set_printoptions(precision=2)
    # we will save the data here
    dir =  os.path.dirname(os.path.realpath(__file__))
    f = h5py.File(os.path.join(dir,"simulation_data"+seria+".hdf5"), "w")
    f.create_dataset('wave', data=wave)
    f.create_dataset('q_abs', data=q_abs)
    f.create_dataset('c_extind', data=c_extind)     
    f.create_dataset('c_absind', data=c_absind)    
    f.create_dataset('field_enh', data=field_enh)
    f.create_dataset('q_ext', data=q_ext)
    f.create_dataset('q_scat', data=q_scat)
    f.create_dataset('p_calc', data = p_calc)
    f.create_dataset('pos', data = pos)
    f.create_dataset('r_eff', data = r_eff)
    f.close()
    plt.plot(wave,c_extind[1,:])
    plt.show()
