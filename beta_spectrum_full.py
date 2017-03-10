# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:50:23 2017

@author: titov
"""

###ЗАМЕТКИ
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

# Constants
ALPHA=1./137.036

PROTON_MASS = 938.2726
NEUTRON_MASS = 939.565
NUCLEON_MASS = 0.5*(PROTON_MASS + NEUTRON_MASS)
ELECTRON_MASS = 0.511
mu_V=4.706 #nucleon magnetic moment difference
LAMBDA=1.2748
PI = math.pi

B_COEFF = [[0.115,-1.8123,8.2498,-11.223,-14.854,32.086],
           [-0.00062,0.007165,0.01841,-0.53736,1.2691,-1.5467],
           [0.02482,-0.5975,4.84199,-15.3374,23.9774,-12.6534],
           [-0.14038,3.64953,-38.8143,172.1368,-346.708,288.7873],
           [0.008152,-1.15664,49.9663,-273.711,657.6292,-603.7033],
           [1.2145,-23.9931,149.9718,-471.2985,662.1909,-305.6804],
           [-1.5632,33.4192,-255.1333,938.5297,-1641.2845,1095.358]]
           
#Auxiliary functions
# Nuclear radius
def radius(A):
    return 0.0029*A**(1./3)+0.0063*A**(-1./3)-0.017*A**(-1)

# Relativistic factor (gamma)
def gamma_rel(Z):
    return math.sqrt(1-(ALPHA*Z)**2)

#
def beta_rel(W):
    E = float(W*ELECTRON_MASS)
    p = math.sqrt(E**2 - ELECTRON_MASS**2)
    return p/E
    
# Coefficients for e/m corrections
def a_coeff(Z):
    A = [0,0,0,0,0,0,0]
    for i in range(0,7):
        for j in range(0,6):
            A[i] = A[i] + B_COEFF[i][j]*(ALPHA*Z)**(j+1)
    return A        

# Dilogarithm function
def dilog(x):
    result = x + 0.25*x**2 + (1.0/9)*x**3 + (1.0/16)*x**4  +0.04*x**5
    return result

# Complex Gamma function
def complexGamma(z):
    result = 0
    n=10000
    result = complex(n**(z))/z
    for i in range (1,n+1):
        #up = up*i**2
        #down = down/((gamma+i)**2 + y**2)
        result = result*i/(z+i)
    #result = up/down
    return result

# Normalizing spectrum to 1
def normalize(x_values, y_values):
    integral = 0
    for i in range(1,len(x_values)):
        integral = integral + (y_values[i]+y_values[i-1])/2*(x_values[i]-x_values[i-1])
    print integral
    normalized_values = []    
    for item in y_values:
        normalized_values.append(item/float(integral))
    result = normalized_values
    return result
    
#class Nucleus:
#class Decay:
    
###############
# Corrections
# Electromagnetic finite-size correction
def L0(Z,A,W):
    rad = radius(A)
    gamma = gamma_rel(Z)
    coef_sum = 0
    for i in range(1,7): 
        coef_sum = coef_sum + a_coeff(Z)[i]*(W*rad)**(i-1)
    
    result = 1 + (13.0/60)*(ALPHA*Z)**2 - W*rad*ALPHA*Z*(41-26*gamma)/(15.0*(2*gamma-1))-ALPHA*Z*rad*gamma*(17.-2*gamma)/(30.0*W*(2*gamma-1)) + a_coeff(Z)[0]*rad/W + coef_sum + 0.41*(rad-0.0164)*(ALPHA*Z)**(4.5)
    #print (13.0/60)*(ALPHA*Z)**2, - W*rad*ALPHA*Z*(41-26*gamma)/(15.0*(2*gamma-1)), -ALPHA*Z*rad*gamma*(17.-2*gamma)/(30.0*W*(2*gamma-1)), a_coeff(Z)[0]*rad/W, coef_sum, 0.41*(rad-0.0164)*(ALPHA*Z)**4.5
    return result

# Weak finite-size corrections    
# Vector (Fermi transition)
def C_V(Z,A,W, endpoint):
    W0 = endpoint/ELECTRON_MASS + 1
    R = radius(A)
    C_0_V = -(233.0/630)*(ALPHA*Z)**2 - 0.2*(W0*R)**2 - (6.0/35)*W0*R*ALPHA*Z#(2.0/35)*W0*R*ALPHA*Z
    C_1_V = (-13.0/35)*R*ALPHA*Z + (4.0/15)*W0*R**2
    C_2_V = (-4.0/15)*R**2
    C_m1_V = (2.0/15)*gamma_rel(Z)*W0*R**2 + (1.0/70)*gamma_rel(Z)*R*ALPHA*Z
    result = 1 + C_0_V + C_1_V*W + C_2_V*W**2 + C_m1_V/W
    return result

# Axial (Gamow--Teller transition)
def C_A(Z,A,W,endpoint):
    W0 = endpoint/ELECTRON_MASS + 1
    R = radius(A)
    C_0_A = (-233.0/630)*(ALPHA*Z)**2 - 0.2*(W0*R)**2 +(2.0/35)*W0*R*ALPHA*Z
    C_1_A = (-21.0/35)*R*ALPHA*Z + (4.0/9)*W0*R**2
    C_2_A = (-4.0/9)*R**2
    C_m1_A = 0
    result = 1 + C_0_A + C_1_A*W + C_2_A*W**2 + C_m1_A/W
    #print C_0_A, C_1_A*W, C_2_A*W**2, C_m1_A/W
    return result
    
# Outer radiative corrections
# For neutrinos
def G_nu(Z,W,endpoint):
    #W0 = endpoint/ELECTRON_MASS + 1
    #W = W0-W
    if W <=1:
        result = 0
    else:
        beta = beta_rel(W)
        result = 3*math.log(NUCLEON_MASS/ELECTRON_MASS) + 23/4.0 - (8.0/beta)*dilog(2*beta/(1+beta)) + 8*(math.atanh(beta)/beta - 1)*math.log(2*beta*W) + 4*(math.atanh(beta)/beta)*(0.125*(7+3*beta**2)-2*math.atanh(beta))
        result = ALPHA*result/(2*PI) + 1
    return result
    
# For electrons
def G_e(Z,W,endpoint):
    beta = beta_rel(W)
    if W <=1:
        result = 0
    else:
        
        W0=endpoint/ELECTRON_MASS + 1
        result = 3*math.log(NUCLEON_MASS/ELECTRON_MASS)-0.75+4*(math.atanh(beta)/beta - 1)*((W0-W)/(3*W) - 1.5 + math.log(2*(W0+0.01-W))) + 4/beta*dilog(2*beta/(1+beta)) + math.atanh(beta)/beta * (2*(1+beta**2) + (W0-W)**2/(6*W**2)-4*math.atanh(beta))
        result = ALPHA*result/(2*PI) + 1
    return result


# Recoil
# Vector transition
def R_V(A, endpoint, W):
    W0 = endpoint/ELECTRON_MASS+1
    M = A*NUCLEON_MASS/ELECTRON_MASS
    r_0_V = W0**2/(2.*M**2) - 11./(6*M**2)
    r_1_V = W0/(3.*M**2)
    r_2_V = 2./M -4.*W0/(3*M**2)
    r_3_V = 16./(3*M**2)
    result = 1 + r_0_V + r_1_V/W + r_2_V*W + r_3_V*W**2
    return result

# Axial transition
def R_A(A, endpoint, W):
    W0 = endpoint/ELECTRON_MASS+1
    M = A*NUCLEON_MASS/ELECTRON_MASS
    r_0_A = -2*W0/(3*M) - W0**2/(6*M**2)- 77./(18*M**2)
    r_1_A = -2./(3*M) + 7.*W0/(9*M**2)
    r_2_A = 10./(3*M) -28.*W0/(9*M**2)
    r_3_A = 88./(9*M**2)
    result = 1 + r_0_A + r_1_A/W + r_2_A*W + r_3_A*W**2
    #if int(1000*W)==1019:
        #print r_0_A, r_1_A/W, r_2_A*W**2, r_3_A*W**3, result
        #print result
    return result

# Screening
# Screening potential
def V_0(Z):
    N = 1.572 #modeldepending shit, see table in Durero diploma
    result = ALPHA**2 *N*(Z-1)**(4./3)
    #print result
    return result
    
#Screening correction
def S(Z,W):
    W_red = W-V_0(Z)
    #print V_0(Z)
    p_red = np.sqrt(W_red**2 - 1)
    p = np.sqrt(W**2 - 1)
    gamma = gamma_rel(Z)
    if W>V_0(Z) and W_red>1:
        #result = (W_red/W) * (p_red/p)**(2*gamma-1) * np.exp(PI*ALPHA*Z*(W_red/p_red -W/p))*LS.Gamma_Func(ALPHA*Z,W_red/p_red,Z,W*ELECTRON_MASS)/(math.gamma(2*gamma+1))**2 #LS.Gamma_Func(ALPHA*Z,W/p,Z,Ee)
        result = (W_red/W) * (p_red/p)**(2*gamma-1) * np.exp(PI*ALPHA*Z*(W_red/p_red -W/p))*np.abs(sp.gamma(gamma + 1j*ALPHA*Z*W_red/p_red))**2/np.abs(sp.gamma(gamma + 1j*ALPHA*Z*W/p))**2
#        if math.isnan(result):
#            print W
#            print W_red**2
#            print gamma
#            print np.exp(PI*ALPHA*Z*(W_red/p_red -W/p))
#            print LS.Gamma_Func(ALPHA*Z,W_red/p_red,Z,W*ELECTRON_MASS)/(math.gamma(2*gamma+1))**2
        return result
    else:
        return 1
# Weak Magnetism 
def B(W, endpoint, TD):
    #delta = 4./3*(mu_V/NUCLEON_MASS)*LAMBDA**(-1)*ELECTRON_MASS
    #result = 1 + delta*W
    Enu = endpoint - ELECTRON_MASS*W
    pe = ELECTRON_MASS*np.sqrt(W**2 - 1)
    Ee = ELECTRON_MASS *W
    beta = pe/Ee
    if TD==0: 
            result = 2.0/3 *(mu_V-0.5)/(NUCLEON_MASS*LAMBDA)*(Ee*beta**2 - Enu)
    elif TD==10:
            result=0
    elif TD==11 or TD==20:
            result=0.6 *(mu_V-0.5)/(NUCLEON_MASS*LAMBDA)*((pe**2+Enu**2)*(Ee*beta**2 - Enu)+2*beta**2 *Enu*Ee*(Enu-Ee)/3)/(pe**2+Enu**2 - 4*beta*Enu*Ee/3)
#        elif TD==21 or TD==30:
#            Space_factor=pnu**4+pe**4+10/3*(pe*pnu)**2
#        elif TD==31 or TD==40:
#            Space_factor=pnu**6+7*(pnu**4*pe**2+pnu**2*pe**4)+pe**6
    else:
        result = 0
    result = result + 1
    return result
    
# Corrections end
###############

def fermi(Z,W,A):
    gamma = gamma_rel(Z)
    p = np.sqrt(W**2 - 1)
    R = radius(A)
    result = 4*(2*p*R)**(2*gamma-2)*np.exp(PI*ALPHA*Z*W/p)*np.abs(complexGamma(gamma + 1j*ALPHA*Z*W/p))**2/(math.gamma(2*gamma+1))**2
    return result
    
#def beta_spectrum(A,Z,endpoint,decayType,branching,de):
#    for i in range
  
  
#def total_spectrum():
    
