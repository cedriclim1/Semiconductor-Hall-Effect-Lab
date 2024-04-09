import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy as sp
import pandas as pd
import scienceplots
import functions

plt.style.use('science')

def resist_func(r1, r2, d):
    return (np.pi * d)/np.log(2) * (r1 + r2)/2 * 1 / np.cosh(np.log(r1/r2)/2.403)

def average_hall_potentials(pos, neg, zero):
    '''
    Display Average Hall Potentials
    '''

    plt.figure(dpi = 150)
    plt.title("Average Hall Potentials vs Temperature")
    plt.xlabel("$T$ (K)")
    plt.ylabel("$V_H$")
    plt.scatter(pos["Temperature (K)"], pos["Hall Potential Average"], label = "+B", s= 5)
    plt.scatter(neg["Temperature (K)"], neg["Hall Potential Average"], label = "-B", s = 5)
    plt.scatter(zero["Temperature (K)"], zero["Hall Potential Average"], label = "Zero B", s = 5)
    plt.plot(pos["Temperature (K)"], pos["Hall Potential Average"], alpha = .5)
    plt.plot(neg["Temperature (K)"], neg["Hall Potential Average"], alpha = .5)
    plt.plot(zero["Temperature (K)"], zero["Hall Potential Average"], alpha = .5)
    plt.legend()
    plt.grid()

def hall_coeff(pos_B, neg_B, d, pos_inv = 0, neg_inv = 0, plot = True):
    '''
    Computes the Hall Coefficient and displays plot
    '''

    # positive and negative current data:
    # Hall Potential Averaged!
    # Include Zero B
    
    pos_hall_coeff = (pos_B["Hall Potential Average"] * (d))/(pos_B["sample I AC"] * pos_B["B-Field In Tesla"])
    neg_hall_coeff = (neg_B["Hall Potential Average"] * (d))/(neg_B["sample I BD"] * neg_B["B-Field In Tesla"])
    
    if plot:
        print("Hall Coefficient: Room Temp Pos = ", pos_hall_coeff[pos_B[np.isclose(pos_B["Temperature (K)"], 293, atol = 2)].index[0]], " Temp = ", pos_B["Temperature (K)"].iloc[pos_B[np.isclose(pos_B["Temperature (K)"], 293, atol = 2)].index[0]])
        print("Hall Coefficient: Room Temp Neg = ", pos_hall_coeff[neg_B[np.isclose(neg_B["Temperature (K)"], 293, atol = 2)].index[0]], " Temp = ", neg_B["Temperature (K)"].iloc[neg_B[np.isclose(neg_B["Temperature (K)"], 293, atol = 2)].index[0]])
        plt.figure(dpi = 150)
        plt.title("Hall Coefficient vs Temperature")
        plt.scatter(pos_B["Temperature (K)"], pos_hall_coeff, label = "+B", s = 5)
        plt.plot(pos_B["Temperature (K)"], pos_hall_coeff, alpha = .5)
        plt.scatter(neg_B["Temperature (K)"], neg_hall_coeff, label = "-B", s = 5)
        plt.plot(neg_B["Temperature (K)"], neg_hall_coeff, alpha = .5)
        plt.xlabel("$T$ (K)")
        plt.ylabel("$R_H$")
        plt.axvline(x = pos_inv, color = 'red', linestyle = '--', alpha = .6, label = "+ Inversion")
        plt.axvline(x = neg_inv, color = 'blue', linestyle = '--', alpha = .6, label = "- Inversion")
        plt.legend()
        plt.grid()
    else:
        return pos_hall_coeff, neg_hall_coeff

def trans_resistivity():
    ...

def resistivity(pos, neg, zero, d, pos_inv = 0, neg_inv = 0, plot = True):
    '''
    Computes the Resistivity and displays plot
    '''

    R_avg_pos_1 = 1/2 * (pos["Voltage DC"]/pos["sample I AB"] + pos["Voltage -DC"]/pos["sample I -AB"] )
    R_avg_pos_2 = 1/2  * (pos["Voltage BC"]/pos["sample I AD"] + pos["Voltage -BC"]/pos["sample I -AD"])
    
    R_avg_neg_1 = 1/2 * (neg["Voltage DC"]/neg["sample I AB"] + neg["Voltage -DC"]/neg["sample I -AB"] )
    R_avg_neg_2 = 1/2  * (neg["Voltage BC"]/neg["sample I AD"] + neg["Voltage -BC"]/neg["sample I -AD"])

    R_pos = resist_func(R_avg_pos_1, R_avg_pos_2, d)
    R_neg = resist_func(R_avg_neg_1, R_avg_neg_2, d)
    
    if plot:    
        print("Resistivity: Room Temp Pos = ", R_pos[pos[np.isclose(pos["Temperature (K)"], 293, atol = 2)].index[0]], " Temp = ",
              pos["Temperature (K)"].iloc[pos[np.isclose(pos["Temperature (K)"], 293, atol = 2)].index[0]])
        print("Resistivity: Room Temp Neg = ", R_neg[neg[np.isclose(neg["Temperature (K)"], 293, atol = 2)].index[0]], " Temp = ",
              neg["Temperature (K)"].iloc[neg[np.isclose(neg["Temperature (K)"], 293, atol = 2)].index[0]])
        plt.figure(dpi = 150)
        plt.title("Resistivity vs Inverse Temperature")
        plt.xlabel("$1/T$ (1/K)")
        plt.ylabel(r"$\rho$")
        
        plt.scatter(1/pos["Temperature (K)"], R_pos, s = 5, label = "$+B$")
        plt.plot(1/pos["Temperature (K)"], R_pos, alpha = .5)
        
        plt.scatter(1/neg["Temperature (K)"], R_neg, s = 5, label = "$-B$")
        plt.plot(1/neg["Temperature (K)"], R_neg, alpha = .5)
        
        plt.axvline(x = 1/pos_inv, color = 'red', linestyle = '--', alpha = .6, label = "+ Inversion")
        plt.axvline(x = 1/neg_inv, color = 'blue', linestyle = '--', alpha = .6, label = "- Inversion")
        plt.legend()
        plt.grid()

        plt.figure(dpi = 150)
        plt.title("Resistivity vs Temperature")
        plt.xlabel("$T$")
        plt.ylabel(r"$\rho$")
        plt.scatter(pos["Temperature (K)"], R_pos, s = 5, label = "$+B$")
        plt.plot(pos["Temperature (K)"], R_pos, alpha = .5)
        plt.scatter(pos["Temperature (K)"], R_neg, s = 5, label = "$+B$")
        plt.plot(pos["Temperature (K)"], R_neg, alpha = .5)
        plt.axvline(x = pos_inv, color = 'red', linestyle = '--', alpha = .6, label = "+ Inversion")
        plt.axvline(x = neg_inv, color = 'blue', linestyle = '--', alpha = .6, label = "- Inversion")
        plt.legend()
        plt.grid()
        
    else:
        return R_pos, R_neg
    

def conductivity(pos, neg, zero, d, pos_inv = 0, neg_inv = 0, plot = True):
    '''
    Computes the conductivity and displays plot
    '''

    conductivity_pos, conductivity_neg = resistivity(pos, neg, zero, d, plot = False)
    
    conductivity_pos = 1/conductivity_pos
    conductivity_neg = 1/conductivity_neg
    
    if plot:
        plt.figure(dpi = 150)
        plt.title("Conductivity vs Inverse Temperature")
        
        plt.scatter(1/pos["Temperature (K)"], conductivity_pos, s = 5, label = "$+B$")
        plt.plot(1/pos["Temperature (K)"], conductivity_pos, alpha = .5)
        
        plt.scatter(1/neg["Temperature (K)"], conductivity_neg, s = 5, label = "$-B$")
        plt.plot(1/neg["Temperature (K)"], conductivity_neg, alpha = .5)
        
        
        plt.ylabel("$\sigma$")
        plt.xlabel("$1/T$ (1/K)")
        plt.xscale('log')
        plt.yscale('log')
        plt.axvline(x = 1/pos_inv, color = 'red', linestyle = '--', alpha = .6, label = "+ Inversion")
        plt.axvline(x = 1/neg_inv, color = 'blue', linestyle = '--', alpha = .6, label = "- Inversion")
        plt.grid()
        plt.legend()
    else:
        return conductivity_pos, conductivity_neg

def carrier_mobility(pos, neg, zero, d, pos_inv = 0, neg_inv = 0, plot = True):

    conductivity_pos, conductivity_neg = conductivity(pos, neg, zero, d, plot = False)
    pos_hall, neg_hall = hall_coeff(pos, neg, d, plot = False)

    if plot:
        plt.figure(dpi = 150)
        plt.scatter(pos["Temperature (K)"], conductivity_pos * pos_hall, label = "+B", s = 5)
        plt.scatter(neg["Temperature (K)"], conductivity_neg * neg_hall, label = "-B", s = 5)
        plt.plot(pos["Temperature (K)"], conductivity_pos * pos_hall, alpha = .5)
        plt.plot(neg["Temperature (K)"], conductivity_neg * neg_hall, alpha = .5)
        plt.title("Carrier Mobility vs Temperature")
        plt.xlabel("Temperature (K)")
        plt.ylabel("$\mu$")
        plt.axvline(x = pos_inv, color = 'red', linestyle = '--', alpha = .6, label = "+ Inversion")
        plt.axvline(x = neg_inv, color = 'blue', linestyle = '--', alpha = .6, label = "- Inversion")
        plt.legend()
        plt.grid()
    else:
        return conductivity_pos * pos_hall, conductivity_neg * neg_hall
def carrier_concentration(pos, neg, zero, d, pos_inv = 0, neg_inv = 0):

    pos_hall, neg_hall = hall_coeff(pos, neg, d, plot = False)
    
    plt.figure(dpi = 150, figsize = (8, 5))
    plt.title("Hole Concentration vs Temperature")
    plt.xlabel("$T$ (K)")
    plt.ylabel("$p$")
    plt.scatter(pos["Temperature (K)"], 1/(1.6*10**(-19) * pos_hall), label = "+B", s = 5)
    plt.scatter(neg["Temperature (K)"], 1/(1.6*10**(-19) * pos_hall), label = "-B", s = 5)
    plt.plot(pos["Temperature (K)"], 1/(1.6*10**(-19) * pos_hall), alpha = .5)
    plt.plot(neg["Temperature (K)"], 1/(1.6*10**(-19) * pos_hall), alpha = .5)
    plt.axvline(x = pos_inv, color = 'red', linestyle = '--', alpha = .6, label = "+ Inversion")
    plt.axvline(x = neg_inv, color = 'blue', linestyle = '--', alpha = .6, label = "- Inversion")
    plt.xlim(95, pos_inv)
    plt.ylim(10**(18), 2.5 * 10**(19))
    # plt.yscale('log')
    plt.legend()
    plt.grid()

# def hall_coeff_cond(pos, neg, zero, d):

#     '''
#     Isn't this just the carrier mobility?
#     '''
#     pos_hall, neg_hall = hall_coeff(pos, neg, d, plot = False)
#     curr_conductivity = conductivity(pos, neg, zero, d, plot = False)
#     hall_cond_pos = curr_conductivity * pos_hall
    
#     plt.figure(dpi = 150)
#     plt.title("Hall Coefficient x Conductivity")
#     plt.xlabel("Temperature (K)")
#     plt.ylabel("R_H * Cond")
#     plt.scatter(pos["Temperature (K)"], hall_cond_pos, s = 5, label = "+B")
#     plt.plot(pos["Temperature (K)"], hall_cond_pos)
#     plt.grid()

def find_inversion_temp(pos, neg, zero, d):
    '''
    Finding the inversion temp via Hall Mobility log-log plot as discussed in Melissinos
    '''

    pos_mob, neg_mob = carrier_mobility(pos, neg, zero, d, plot = False)

    # Positive Fit

    pos_last_ind = pos_mob[pos_mob < 0].index[0]
    pos_first_ind = pos_last_ind - 1
    
    fit = np.polyfit([pos["Temperature (K)"].iloc[pos_first_ind], pos["Temperature (K)"].iloc[pos_last_ind]], [pos_mob.iloc[pos_first_ind], pos_mob.iloc[pos_last_ind]], 1)
    pos_theory_0 = np.abs(fit[1]/fit[0])
    
    # Negative Fit

    neg_last_ind = neg_mob[neg_mob < 0].index[0]
    neg_first_ind = neg_last_ind - 1

    fit = np.polyfit([neg["Temperature (K)"].iloc[neg_first_ind], neg["Temperature (K)"].iloc[neg_last_ind]], [neg_mob.iloc[neg_first_ind], neg_mob.iloc[neg_last_ind]], 1)
    neg_theory_0 = np.abs(fit[1]/fit[0])
    
    return pos_theory_0, neg_theory_0


def b_fitting(pos, neg, zero, d, plot = True):
    
    '''
    Determines the b ratio by fitting a linear fit on log log scale
    '''
    
    
    pos_inv, neg_inv = find_inversion_temp(pos, neg, zero, d)
    pos_inv_log = np.log(1/pos_inv)
    neg_inv_log = np.log(1/neg_inv)
    
    R_pos, R_neg = resistivity(pos, neg, zero, d, plot = False)
    
    # Log-Log Scaling
    
    pos_log_t = np.log(1/pos["Temperature (K)"])
    R_pos_log = np.log(R_pos)
    
    neg_log_t = np.log(1/neg["Temperature (K)"])
    R_neg_log = np.log(R_neg)
    
    # Fitting
    
    R_pos_fit = np.polyfit(pos_log_t[:30], R_pos_log[:30], deg = 1)
    R_pos_predict = pos_log_t * R_pos_fit[0] + R_pos_fit[1]
    
    R_neg_fit = np.polyfit(neg_log_t[:30], R_neg_log[:30], deg = 1)
    R_neg_predict = neg_log_t * R_neg_fit[0] + R_neg_fit[1]
    
    if plot:
        plt.figure(dpi = 150, figsize = (7, 5))
        plt.title(r"Log-Log Plot of $\rho$ and $1/T$")
        plt.xlabel(r"$\log{1/T}$")
        plt.ylabel(r"$\log{\rho}$")
        plt.scatter(pos_log_t, R_pos_log, s = 5, label = "$+B$")
        plt.scatter(neg_log_t, R_neg_log, s = 5, label = "$-B$")

        # Fits
        plt.plot(pos_log_t, R_pos_predict, alpha = .5, label = "$+B$ Fit")
        plt.plot(neg_log_t, R_neg_predict, alpha = .5, label = "$-B$ Fit")
        # Inverison Temps

        plt.axvline(pos_inv_log, alpha = .5, label = "Theoretical $+B$ Inversion Temp", color = 'red')
        plt.axvline(neg_inv_log, alpha = .5, label = "Theoretical $-B$ Inversion Temp", color = 'orange')

        plt.grid()
        plt.legend()
    else:
        
        pos_closest_t = pos_log_t[pos_log_t <= pos_inv_log].iloc[0]
        neg_closest_t = neg_log_t[neg_log_t <= neg_inv_log].iloc[0]
        
        pos_closest_t_ind = pos_log_t[pos_log_t <= pos_inv_log].index[0]
        neg_closest_t_ind = neg_log_t[neg_log_t <= neg_inv_log].index[0]
        
        
        b_pos = np.exp(pos_closest_t * R_pos_fit[0] + R_pos_fit[1])/(np.exp(pos_closest_t * R_pos_fit[0] + R_pos_fit[1]) - (R_pos[pos_closest_t_ind]))
        b_neg = np.exp(neg_closest_t * R_neg_fit[0] + R_neg_fit[1])/(np.exp(neg_closest_t * R_neg_fit[0] + R_neg_fit[1]) - R_neg[neg_closest_t_ind])

        return b_pos, b_neg

def hole_electron_mobilities_extrinsic(pos, neg, zero, d):
    
    b_pos, b_neg = b_fitting(pos, neg, zero, d, plot = False)
    pos_mob, neg_mob = carrier_mobility(pos, neg, zero, d, plot = False)
    pos_inv, neg_inv = find_inversion_temp(pos, neg, zero, d)
    
    pos_mob_e = b_pos * pos_mob
    neg_mob_e = b_neg * neg_mob
    
    plt.figure(dpi = 150, figsize = (5, 3))
    plt.scatter(pos["Temperature (K)"], pos_mob, label = "+B", s = 5)
    plt.scatter(neg["Temperature (K)"], neg_mob, label = "-B", s = 5)
    plt.scatter(pos["Temperature (K)"], pos_mob, alpha = .5, label = r"$\mu_{+,H}$", s = 5)
    plt.scatter(neg["Temperature (K)"], neg_mob, alpha = .5, label = r"$\mu_{-,H}$", s = 5)
    plt.scatter(pos["Temperature (K)"], pos_mob_e, alpha = .5, label = r"$\mu_{+, e}$", s = 5)
    plt.scatter(neg["Temperature (K)"], neg_mob_e, alpha = .5, label = r"$\mu_{-, e}$", s = 5)
    plt.title("Carrier Mobility in Extrinsic Region")
    plt.xlabel("Temperature (K)")
    plt.ylabel("$\mu$")
    # plt.xscale('log')
    plt.yscale('log')
    # plt.axvline(x = pos_inv, color = 'red', linestyle = '--', alpha = .6, label = "+ Inversion")
    # plt.axvline(x = neg_inv, color = 'blue', linestyle = '--', alpha = .6, label = "- Inversion")
    plt.xlim(95, neg_inv - 20)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.grid()
    
def form_set(data):

    '''
    Average Datasets Hall Potential
    '''
    
    data["B-Field In Tesla"] = data["B-Field (Gauss)"]/10000
    Negative_B = data.iloc[::3, :].reset_index()
    Zero_B = data.drop(0).iloc[::3, :].reset_index()
    Positive_B = data.drop([0, 1]).iloc[::3, :].reset_index()

    Positive_B["Hall Potential Average"] = -1/4 * (Positive_B["Voltage AC"] - Positive_B["Voltage -AC"] + Positive_B["Voltage BD"] - Positive_B["Voltage -BD"])
    Negative_B["Hall Potential Average"] = -1/4 * (Negative_B["Voltage AC"] - Negative_B["Voltage -AC"] + Negative_B["Voltage BD"] - Negative_B["Voltage -BD"])
    Zero_B["Hall Potential Average"] = 1/4 * (Zero_B["Voltage AC"] - Zero_B["Voltage -AC"] + Zero_B["Voltage BD"] - Zero_B["Voltage -BD"])
    
    return Positive_B, Negative_B, Zero_B
    

def gen_data_plots(data):
    '''
    Generates all plots in one go including:
    Hall Coefficient, Resistivity, Conductivity, Hall Coefficient * Conductivity,
    Hall Coefficient and Hall Mobility vs Temperature.
    '''
    # ''Depth'' of the Germanium sample
    d = .00125

    # For each individual B-Field compute the average Hall Potential
    
    Positive_B, Negative_B, Zero_B = form_set(data)
    pos_inv, neg_inv = find_inversion_temp(Positive_B, Negative_B, Zero_B, d)
    print(pos_inv, neg_inv)
    average_hall_potentials(Positive_B, Negative_B, Zero_B)
    hall_coeff(Positive_B, Negative_B, d, pos_inv = pos_inv, neg_inv = neg_inv)
    resistivity(Positive_B, Negative_B, Zero_B, d, pos_inv = pos_inv, neg_inv = neg_inv)
    conductivity(Positive_B, Negative_B, Zero_B, d, pos_inv = pos_inv, neg_inv = neg_inv)
    carrier_mobility(Positive_B, Negative_B, Zero_B, d, pos_inv = pos_inv, neg_inv = neg_inv)
    carrier_concentration(Positive_B, Negative_B, Zero_B, d, pos_inv = pos_inv, neg_inv = neg_inv)
    b_fitting(Positive_B, Negative_B, Zero_B, d)
    hole_electron_mobilities_extrinsic(Positive_B, Negative_B, Zero_B, d)
    
    # Just carrier mobility?
    # hall_coeff_cond(Positive_B, Negative_B, Zero_B, d)