import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from bubble import get_bubble_radius

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[0] = 0
    return y_smooth

def get_force(Rmax, r_range_max, dQdt,H,rho_oil,gravity,gamma,T, pre_jump):
    P_inf = 101325
    N_t = np.arange(1, len(T) + 1)
    r_range = np.linspace(0, r_range_max, int(r_range_max / (0.01 * Rmax)))
    u = np.zeros((len(T), len(r_range)))
    Pp1 = rho_oil * gravity * H * np.ones_like(u)
    Pp2 = np.zeros_like(u)
    Pp3 = np.zeros_like(u)
    Pp = np.zeros_like(u)
    Ppb = np.zeros_like(u)
    Fz = np.zeros_like(T)
    for i4 in range(len(N_t)):
        for ir in range(len(r_range)):
            Pp2[i4, ir] = rho_oil * dQdt[N_t[i4]-1 ] / (4 * np.pi * gamma)
            Pp[i4, ir] = Pp1[i4, ir] + Pp2[i4, ir] + Pp3[i4, ir]
            Ppb[i4, ir] =Pp1[i4, ir] + Pp2[i4, ir] + Pp3[i4, ir]
            if pre_jump == True:
                Fz[i4] = 2 * np.pi * trapz(r_range, r_range * Ppb[i4, :]) # positive sign due to Newton's 3rd law
            else:
                Fz[i4] = -2 * np.pi * trapz(r_range, r_range * Pp[i4, :])  
    return Fz        

def beaker_dynamics(pre_jump,time, bubble_rad,jump_idx,exp_idx, H,rho_oil,gravity,gamma,m_beaker,m_oil,u0,box,dt):
    R_data =bubble_rad
    T_data = time
    Rmax = np.max(R_data)
    L = gamma * np.max(R_data)
    T = T_data
    # Currently using moving average smoothing function with box=7, while the Matlab code use LOESS. I will implement LOESS here as well if necessary
    R_b = smooth(R_data,box)    
    R_b[-box:] = R_data[-box:]
    dR_bdt = np.gradient(R_b, T)
    dR_bdt = smooth(dR_bdt,box)
    dR_bdt[-box:] = dR_bdt[-box:]
    d2Rdt2 = np.gradient(dR_bdt, T)
    d2Rdt2 = smooth(d2Rdt2,box)
    d2Rdt2[-box:] = d2Rdt2[-box:]
    Q = 4 * np.pi * (R_b ** 2) * dR_bdt
    dQdt = 4 * np.pi * (2 * R_b * (dR_bdt ** 2) + (R_b ** 2) * d2Rdt2)
    r_range_max = 30.4e-3
    M_tot = m_beaker + m_oil  # Total mass [kg]
    Mg = M_tot* gravity  # Gravitational force on the beaker [N]
    
    if pre_jump == False:
        Fz = get_force(Rmax,r_range_max,dQdt,H,rho_oil,gravity,gamma,T, pre_jump)
        v_z = np.zeros_like(Fz)  # Initial velocity array
        a_z = np.zeros_like(Fz)  # Initial acceleration array
    
        v_z[jump_idx] = u0
        

        tplot = np.linspace(0,len(Fz),len(Fz))*dt
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.figure()
        plt.plot(tplot[jump_idx:]*1000,(Fz[jump_idx:])/Mg,linewidth=2)
        plt.xlabel('Time (ms)',fontsize=60)
        plt.ylabel(r'$F_{b} / F_g$',fontsize=60)
        #plt.ylim(0,50)
        plt.tick_params(axis='both', which='major', labelsize=32)
        plt.tick_params(axis='both', which='minor', labelsize=32)
        #plt.legend(loc="best",prop={'size': 30})
        plt.tight_layout()
        plt.savefig("force_jump.png")

        # Calculating acceleration and velocity after the bubble impacts the bottom of the beaker
        for i in range(jump_idx+1, len(Fz)):
            a_z[i] =  (Fz[i] - Mg) / M_tot
            v_z[i] = v_z[i-1] + trapz([a_z[i-1], a_z[i]], dx=dt)
        # Calculating displacement over time
    
        D_z = np.zeros_like(Fz)
        for i in range(1, len(D_z)):
            D_z[i] = D_z[i-1] + trapz([v_z[i-1], v_z[i]], dx=dt)
    
        return D_z
    
    elif pre_jump == True:
        Fz = get_force(Rmax,r_range_max,dQdt,H,rho_oil,gravity,gamma,T, pre_jump)
        tplot = np.linspace(0,len(Fz),len(Fz))*dt
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.figure()
        plt.plot(tplot[:(jump_idx)]*1000,(Fz[:(jump_idx)])/Mg,linewidth=2)
        plt.xlabel('Time (ms)',fontsize=60)
        plt.ylabel(r'$F_{gr} / F_g$',fontsize=60)
        plt.ylim(0,50)
        plt.tick_params(axis='both', which='major', labelsize=48)
        plt.tick_params(axis='both', which='minor', labelsize=48)
        #plt.legend(loc="best",prop={'size': 30})
        plt.tight_layout()
        plt.savefig("bottom_force.png")
        impulse1 = np.sum(Fz[:jump_idx]- Mg)*dt
        u01 = impulse1 / M_tot
        #print(u01)
        return u01

if __name__ == '__main__':

    #u0_model=  beaker_dynamics(pre_jump=True,time=T_data,bubble_rad=R_data,jump_idx=jump_idx, exp_idx=exp_idx,H=H,rho_oil=rho_oil,gravity=gravity,gamma=gamma,m_beaker=m_beaker,m_oil=m_oil,u0=0,box=box,dt=dt)
    
    #h_model =  beaker_dynamics(pre_jump=False,time=T_data,bubble_rad=R_data,jump_idx=jump_idx, exp_idx=exp_idx,H=H,rho_oil=rho_oil,gravity=gravity,gamma=gamma,m_beaker=m_beaker,m_oil=m_oil,u0=u0_model,box=box,dt=dt)

    ##material parameters for bubble motion##
    drop_rad = 3e-3 # (m)
    sigma = 5e-2 #surface tension (N/m)
    oil_temp = 150 + 273.15 #(K)
    water_temp = 30 + 273.15 #(K)
    rho_l = 800 # (kg / m3) density of paraffin oil
    c_pl = 2181 # (J kg-1 K-1) heat capacity of paraffi oil
    rho_v = 2.  # (kg / m3) density of water vapor (at 150degC)
    k_l = 0.1   #(W/m K) heat conductivity of paraffin oil
    latent_heat = 2.26e6 #(J/kg) latent heat of vaporization of water
    delta_T = oil_temp - (100+273.15) 
    t_param = rho_l,c_pl, rho_v,k_l, latent_heat, delta_T

    ##Fitting parameters for bubble motion##
    alpha = 0.36 #fitting parameter representing the amount of droplet that evaporate during explosions
    visc = 50 # damping constant for the oscillation part
    omega = 1900 # oscillation frequency
    time = np.linspace(0,0.0124,125)
    dt = 1e-4
    mass_const = 0.0016# fitting parameter to determine the evaporating mass flux
    beta = 0.042
    explosion_time = 23 # transition from diffusion to explosion (currently a fitting param)

    osc_param = visc,omega

    r_bubble, jump_idx = get_bubble_radius(time,texp_ind=explosion_time,drop_rad=alpha*drop_rad,thermo_param=t_param,osc_param=osc_param, mass_flux_fit=mass_const,beta=beta,sigma=sigma,dt=dt)

    ##parameters for beaker motion
    gravity = 9.8
    H = 51e-3  # Depth of oil
    m_beaker = 0.194  # Mass of the beaker [kg]
    m_oil = 0.14     # oil mass [kg]
    gamma = 0.0023  # distance between bubble and the beaker floor (currently fitting parameter)

    box = 7 #smoothing box (will be replaced once we used LOESS for smoothing function)

    Data = np.loadtxt("./data_1.txt",skiprows=1)

    T_data = Data[:, 0] * 1e-3
    R_data = (3 / 4 / np.pi) ** (1 / 3) * (Data[:, 1] * (1e-3) ** 3) ** (1 / 3)
    h_data = Data[:,2]  


    u0_model=  beaker_dynamics(pre_jump=True,time=T_data,bubble_rad=r_bubble, jump_idx=jump_idx, exp_idx=explosion_time,H=H,rho_oil=rho_l,gravity=gravity,gamma=gamma,m_beaker=m_beaker,m_oil=m_oil,u0=0,box=box,dt=dt)
    print("u0=" + str(u0_model))
    h_model =  beaker_dynamics(pre_jump=False,time=T_data,bubble_rad=r_bubble,jump_idx=jump_idx, exp_idx=explosion_time,H=H,rho_oil=rho_l,gravity=gravity,gamma=gamma,m_beaker=m_beaker,m_oil=m_oil,u0=u0_model,box=box,dt=dt)
    
    #Plotting bubble radius over time
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(T_data*1000,r_bubble*1000,label="Model",linewidth=2)
    plt.plot(T_data*1000, R_data*1000,"-s",label="Experiment",linewidth=2,markersize=5)
    plt.xlabel('Time (ms)',fontsize=60)
    plt.ylabel('Bubble radius(mm)',fontsize=60)
    plt.xlim(0,13)
    #plt.title('Bubble radius vs Time')
    #plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.tick_params(axis='both', which='minor', labelsize=48)
    plt.legend(loc="best",prop={'size': 50},framealpha=0.5)
    plt.tight_layout()
    plt.savefig("bubble_radius.png")
    
    #Plotting displacement over time
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(T_data*1000,h_model*1000,label="Model",linewidth=2)
    plt.plot(T_data*1000, h_data,"-s",label="Experiment",linewidth=2,markersize=5)
    plt.xlabel('Time (ms)',fontsize=60)
    plt.ylabel('Displacement (mm)',fontsize=60)
    plt.xlim(0,13)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.tick_params(axis='both', which='minor', labelsize=48)
    plt.legend(loc="best",prop={'size': 50},framealpha=0.5)
    plt.tight_layout()
    plt.savefig("beaker_jump.png")
  