import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import pandas as pd 
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def simulate_sinking_sphere(r,H):
    # Define the differential equation for the sinking sphere (for visualizing sinking droplet)
    def sphere_diff_eq(t, y, rho_s, rho_w, r):
        V = (4/3) * np.pi * r**3
        F_buoyancy = rho_w * V * g
        F_drag = 6 * np.pi * mu_oil * r * y[1]
        dydt = [y[1], (F_buoyancy - rho_s * V * g - F_drag) / (rho_s * V)]
        return dydt

    rho_w = 1.01  
    rho_oil = 0.9 
    #r = 0.1  
    g = 9.81  # Acceleration due to gravity (m/s^2)
    mu_oil = 1  
    t_span = [0,100]  # Time span for simulation
    y0 = [0, 0]  # Initial conditions [position, velocity]
  
    sol = solve_ivp(sphere_diff_eq, t_span, y0, args=(rho_w, rho_oil, r))
    depth = sol.y[0]
    depth = np.where(depth < -H,-H,depth)
    b = depth + H 
    t_idx = np.where(b == b.min())
    time = sol.t
    falltime = time[t_idx[0]]

    return depth,falltime

def fourier_solution(a,r,t,alpha,N):  # t is float
    sol = np.zeros(r.shape)
    for n in range(1,N):
        sol_n = ((2*a)/(np.pi*r)) * ((((-1)**n) /n) * np.sin((n*np.pi*r)/a) * np.exp((-alpha*(n**2)*(np.pi**2)*t )/(a**2)) )
        sol += sol_n
    return sol

def get_evaporation_time(droplet_rad,t_arr,alpha,N):
    # to estimate time when dr
    r = np.linspace(1e-3,droplet_rad,100)
    step = 0
    deltat = 0.1
    t0 = 0
    cutoff = 1e-3
    while t0 < 1-cutoff:
        t = step * deltat
        sol = np.ones(r.shape) + fourier_solution(droplet_rad,r,t,alpha,N)
        #print(sol[0])
        t0 = sol[0]
        step +=1
    print(step*deltat)
    return step*deltat
  
def get_osc_time(rdrop,beta,uf,dt):
    #function to get the time when all the liquid vaporize , see notes for derivation
    tosc = (1/3) * (rdrop / (beta*uf))
    return int(tosc/dt)

def get_jakob(param):
    #function to calculate Jakob number from various liquid property
    rho_l,c_pl, rho_v,k_l, latent_heat, delta_T = param
    #Ja = (rho_l * c_pl * delta_T) / (rho_v * latent_heat) # I dont know which definition Ja is appropriate but the bottom one works better so yeah.
    Ja= ( c_pl * delta_T) / (latent_heat)
    return Ja

def get_diff(param):
    # function to calculate liquid thermal diffusivity
    rho_l,c_pl, rho_v,k_l, latent_heat, delta_T = param
    diff = k_l / (rho_l / c_pl)
    return diff

def solve_diffusion(r,Ja, D_l,deltat):
    # function to calculate the thermally controlled growth of vapor bubble 
    t = np.linspace(0,len(r),len(r)) * deltat
    print(Ja)
    r =  2 * Ja * np.sqrt((3 * D_l * t) / np.pi)
    return r

def get_mass_flux(param,sigma,fit):
    # function to calculate mass flux (speed)
    rho_l,c_pl, rho_v,k_l, latent_heat, delta_T = param
    uf = (fit*delta_T / sigma) #- (b/sigma) #*rho_l or rho_v
    return uf

def solve_explosion(r,r0,uf,thermo_param,deltat):
    # function to calculate bubble radius during explosion 
    rho_l,c_pl, rho_v,k_l, latent_heat, delta_T = thermo_param
    t = np.linspace(0,len(r),len(r)) * deltat
    r = r0 + (r0/4)*((uf*(t)/(r0))**3)
    return r

def solve_osc(r,r0,req,rp,deltat):
    # function to calculate bubble radius during oscillation
    t = np.linspace(0,len(r),len(r)) * deltat
    eta,omega = rp
    def osc_diff_eq(X, t, b,o):
        x, dotx = X
        ddotx =  -2*b*dotx - (o**2)*x
        return [dotx,ddotx]
    x0 = (r0/req) - 1  
    init= [x0,0]
    sol = odeint(osc_diff_eq,init,t,args = (eta,omega) )
    x_t = sol[:,0]
    r = req*(1 + x_t)
    return r

def get_eq_rad_plesset(drop_rad,thermo_param): #time when the droplet completely vaporized using classical Plesset Zwick theory (thermodynamic equilibrium)
    rho_w = thermo_param[0]
    rho_v = thermo_param[2]
    m0 = (4/3)*np.pi*(drop_rad**3)*rho_w
    jakob = get_jakob(thermo_param) # Jakob number
    d_l = get_diff(thermo_param) # liquid thermal diffusivity
    Lamda = 64*(jakob**3)*np.sqrt(3/np.pi)*(d_l**1.5)*rho_v
    tstop = (m0/Lamda)**(2/3)
    gas_vol = m0 / rho_v
    gas_rad = (3 * gas_vol / (4 * np.pi))**(1/3) 
    r_eq =  2 * jakob * np.sqrt((3 * d_l * tstop) / np.pi)
    return r_eq

def get_bubble_radius(t_arr,texp_ind,drop_rad,thermo_param, osc_param,mass_flux_fit,beta,sigma,dt):
    r = np.zeros(t_arr.shape)    
    jakob = get_jakob(thermo_param) # Jakob number
    d_l = get_diff(thermo_param) # liquid thermal diffusivity
    mass_flux = get_mass_flux(thermo_param,sigma,mass_flux_fit)
    tosc_ind = get_osc_time(drop_rad,beta,mass_flux,dt) 
    print(tosc_ind)
    r_dif = r[:texp_ind]
    r_dif = solve_diffusion(r_dif,jakob,d_l,dt)
    r0_exp = r_dif[-1] 
    r_exp = r[texp_ind:tosc_ind]
    r_exp = solve_explosion(r_exp,r0_exp,mass_flux,thermo_param,dt)
    r0_osc = r_exp[-1]
    r_osc = r[tosc_ind:]
    req = get_eq_rad_plesset(drop_rad,thermo_param)
    print(req*1000)
    r_osc = solve_osc(r_osc,r0_osc,req,osc_param,dt)
    r_all = np.concatenate((r_dif, r_exp, r_osc))
    #plt.plot(t_arr,r_all)
    #plt.show()
    return r_all, tosc_ind+1 #assuming beaker jump right after explosion is over

if __name__ == '__main__':

    ##Material parameters##
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

    ##Fitting parameters##
    alpha = 0.5 #fitting parameter representing the amount of droplet that evaporate during explosions
    visc = 50 # damping constant for the oscillation part
    omega = 2000 # oscillation frequency
    time = np.linspace(0,0.0124,125)
    dt = 1e-4
    mass_const = 0.0015 # fitting parameter to determine the evaporating mass flux
    beta = 0.05
    explosion_time = 23 # transition from diffusion to explosion (currently a fitting param)
    osc_param = visc,omega

    r_bubble, tjump = get_bubble_radius(time,texp_ind=explosion_time,drop_rad=alpha*drop_rad,thermo_param=t_param,osc_param=osc_param, mass_flux_fit=mass_const,beta=beta,sigma=sigma,dt=dt)


