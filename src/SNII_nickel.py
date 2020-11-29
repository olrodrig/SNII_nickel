import numpy as np
from plot import plot_logMNi

class Nickel_mass:

  def __init__(self, band, t_mag, mag, err_mag, properties, tmin=95.0, tmax=320.0, RxG=99, Rxh=99, apply_kcor=True, verbose=True):
    
    if band not in ['V', 'r', 'R', 'i', 'I']:
        print('Error: input band "'+band+'" is not valid!')
        print('       Valid bands: "V", "r", "R", "i", "I".')
    else:
    
        self.band = band
        self.mag = mag
        self.emag = err_mag
        self.logMNi_i  = {}
        self.elogMNi_i = {}
        self.logMNi    = {}
        self.elogMNi   = {}
        self.MNi    = {}
        self.eMNi   = {}
        self.ssd       = {}
        self.tmin = tmin
        self.tmax = tmax
        
        #total-to-selective extinction ratios
        Al_EBV = {'V':2.96, 'r':2.50, 'R':2.33, 'i':1.94, 'I':1.68}
        if RxG == 99:  RxG = Al_EBV[band]
        if Rxh == 99:  Rxh = Al_EBV[band]
        
        #BC calibrations
        if band == 'V':  zp, error_zp, beta = 11.15, 0.18, 0.0
        if band == 'r':  zp, error_zp, beta = 11.89, 0.16, 0.0
        if band == 'R':  zp, error_zp, beta = 12.07, 0.14, 0.0
        if band == 'i':  zp, error_zp, beta = 11.91, 0.12, 0.0
        if band == 'I':  zp, error_zp, beta = 12.37, 0.10, 0.036
        
        #ZP systematic error
        error_alpha = 0.04
        error_zp = np.sqrt(error_zp**2+error_alpha**2)
        
        #check SN properties
        compute = True        
        for q in ['z_helio', 'mu', 'err_mu', 'EGBV', 'err_EGBV', 'EhBV', 'err_EhBV', 'texp', 'err_texp']:
            if q not in properties.keys():
                print('Error: '+q+' not provided')
                compute = False
        
        if compute:
            
            #SN properties
            z_helio = properties['z_helio']
            t0, et0 = properties['texp'], properties['err_texp']
            mu, emu = properties['mu'], properties['err_mu']
            EGBV, eEGBV = properties['EGBV'], properties['err_EGBV']
            EhBV, eEhBV = properties['EhBV'], properties['err_EhBV']
            
            #B term and its error
            B     = (mu - zp + RxG*EGBV + Rxh*EhBV)/2.5 - (0.39 - beta/2.5)*t0/(1.0+z_helio)/100.0 - 3.076
            var_B = (emu**2 + error_zp**2 + (eEGBV*RxG)**2 + (eEhBV*Rxh)**2)/6.25 + ((0.39 - beta/2.5)*et0/(1.0+z_helio)/100.0)**2
            err_B = np.sqrt(var_B)
            self.err_B = err_B
            
            #time since the explosion in the SN rest frame
            Dt = (t_mag-t0)/(1.0+z_helio)
            self.Dt = Dt
            
            #values within [tmin, tmax]
            mask = (Dt>=tmin) & (Dt<=tmax)
            t_i  = t_mag[mask]
            Dt_i = Dt[mask]
            m_i  = mag[mask]
            error_m_i = err_mag[mask]
                    
            N = len(Dt_i)
            self.N = N
            
            if min(Dt_i) <  95.0: print('Warning: %i points at t<95 days since explosion' % len(Dt_i[Dt_i<95.0]))
            if max(Dt_i) > 320.0: print('Warning: %i points at t>320 days since explosion' % len(Dt_i[Dt_i>320.0]))
            if min(Dt_i) <  95.0 or max(Dt_i) > 320.0: print('         BC calibrations are valid in the range 95-320 days since explosion.')
            if min(Dt_i) <  95.0 or max(Dt_i) > 320.0: print('         Consider changing the tmin and/or tmax values.')
            if verbose:  print('%i %s-band photometric points between %5.1f and %-5.1f days since explosion.\n' % (N, band, tmin, tmax))
            
            if apply_kcor:
                K, error_K = Kcor(band, z_helio, Dt_i)
                m_i  = m_i - K
                error_m_i = np.sqrt(error_m_i**2+error_K**2) 
            
            if N == 0:
                print('There is no photometry between 95 and 320 days since explosion.')
            else:    
            
                #A_i estimates
                A_i     = -m_i/2.5 + (0.39 - beta/2.5)*t_i/(1.0+z_helio)/100.0
                err_A_i = error_m_i/2.5
                
                #A value that maximizes the log-likelihood of a constant-only model
                if N == 1:
                    A, ssd_A = A_i[0], err_A_i[0]
                else:
                    A, ssd_A = likelihood_maximization(A_i, err_A_i, n_pars=1)[0:2]
               
                #check if it is necessary to correct for gamma-ray leakage
                if N in [1, 2]:
                    print('Warning: only %i Ni-56 estimates' % (N)) 
                    print('         gamma-ray leakage cannot be checked.')  
                    fdep_needed = False
                    err_A = ssd_A
                else:
                    if verbose:  print('Cheking constancy of the Ni-56 mass estimates...\n')
                    appropriated, pars = IC(Dt_i, A_i, err_A_i, min([1, N-2]), verbose=verbose)
                    if appropriated[0] == 'yes':  
                        if verbose:  print('Ni-56 mass estimates are consistent with a constant value.')
                        err_A = ssd_A/np.sqrt(float(N)) 
                        fdep_needed = False
                    if appropriated[0] == 'no' :
                        err_A = ssd_A  
                        fdep_needed = True
                
                lMNi_i  = A_i + B
                elMNi_i = err_A_i
                lMNi    = A+B
                elMNi   = np.sqrt(err_A**2+err_B**2)
                            
                self.fdep = False
                self.logMNi_i['fdep=1']  = lMNi_i
                self.elogMNi_i['fdep=1'] = elMNi_i
                self.logMNi    = lMNi
                self.elogMNi   = elMNi
                self.MNi       = 10.0**lMNi
                self.eMNi      = np.log(10.0)*elMNi*10.0**lMNi
                self.ssd['fdep=1']       = ssd_A
                self.Dt_i    = Dt_i
                
                if fdep_needed:
                    if pars[1]>0.0:
                        if verbose:  print('Ni-56 mass estimates increase with time.')
                        if verbose:  print('The deposition function is not required.')
                    if pars[1]<0.0:
                        if verbose:  print('It is necessary to include the deposition function (fdep)\n')
                        if verbose:  print('Computing T_0...')
                    
                        AD_i, AD, T_0, ssd_A = fdep_cor(Dt_i, A_i, err_A_i)
                        if verbose:  print('T_0=%i days\n' % int(round(T_0,0)))
                                            
                        if verbose:  print('Cheking constancy of the Ni-56 mass estimates (corrected for fdep)...\n')
                        appropriated = IC(Dt_i, AD_i, err_A_i, min([1, N-2]), verbose=verbose)[0]
                        if appropriated[0] == 'yes': 
                            if verbose:  print('Ni-56 mass estimates are consistent with a constant value.')
                            err_A = ssd_A/np.sqrt(float(N)) 
                        if appropriated[0] == 'no' :
                            if verbose:  print('Ni-56 mass estimates are not consistent with a constant value.')
                            err_A = ssd_A 
                        
                        lMNi_i  = AD_i + B
                        elMNi_i = err_A_i
                        
                        lMNi  = AD+B
                        elMNi = np.sqrt(err_A**2+err_B**2)
                        
                        self.fdep = True
                        self.T_0 = T_0
                        self.ssd['fdep<1']       = ssd_A
                        self.logMNi_i['fdep<1']  = lMNi_i
                        self.elogMNi_i['fdep<1'] = elMNi_i
                        self.logMNi_unc    = self.logMNi.copy()
                        self.elogMNi_unc   = self.elogMNi.copy()
                        self.MNi_unc       = self.MNi.copy()
                        self.eMNi_unc      = self.eMNi.copy()
                        self.logMNi    = lMNi
                        self.elogMNi   = elMNi
                        self.MNi       = 10.0**lMNi
                        self.eMNi      = np.log(10.0)*elMNi*10.0**lMNi
                
    
  def plot(self, panels, sn='', figure_name=''):
    plot_logMNi(panels, self, sn=sn, figure_name=figure_name)
    
def Kcor(band, z_helio, phase):

  if z_helio > 0.043:  
      print('Warning: implemented K-correction is valid for z_helio<0.043 (input z_helio='+str(z_helio)+')')
      print('         Consider providing K-corrected photometry to the SNII_nickel code.\n')
      
  if band == 'V':  K_pars, ssd_K = [8.10, -1.69], 2.97
  if band == 'r':  K_pars, ssd_K = [4.04,  0.92], 2.30
  if band == 'R':  K_pars, ssd_K = [1.48,  0.57], 0.92
  if band == 'i':  K_pars, ssd_K = [1.09,  0.39], 0.94
  if band == 'I':  K_pars, ssd_K = [6.31, -1.69], 1.07
  
  K       = -2.5*np.log10(1.0+z_helio)+z_helio*(K_pars[0]+K_pars[1]*(phase/100.0))
  error_K = z_helio*ssd_K
  
  return K, error_K

def likelihood_maximization(A_i, err_A_i, n_pars=1):

  N = len(A_i)
  if N < 3:
      e0s = [0.0]
  else:
      A = np.mean(A_i)
      residuals = A_i-A
      ssd = np.sqrt(np.sum(residuals**2)/float(N-n_pars))
      e0s = np.linspace(0.0, 2.0*ssd, 201)
  m2lnL_min = 1.e99
  for e0 in e0s:
      Var = err_A_i**2+e0**2
      A = np.sum(A_i/Var)/np.sum(1.0/Var)
      m2lnL = np.sum(np.log(Var)+(A_i-A)**2/Var)
      if m2lnL < m2lnL_min:
          m2lnL_min = m2lnL
          A_min     = A
          e0_min    = e0
  
  if e0_min != 0.0:  n_pars = n_pars + 1
  A, m2lnL = A_min, m2lnL_min
  residuals = A_i-A
  ssd = np.sqrt(np.sum(residuals**2)/float(N-n_pars))
  
  return A, ssd, m2lnL

def fdep_cor(Dt_i, A_i, err_A_i):
    
  T_0s = np.linspace(100.0, 700.0, 601)[::-1]
  
  m2lnL_min = 1.e99
  for T_0 in T_0s:
      D_i  = np.log10(0.97*(1.0-np.exp(-(T_0/Dt_i)**2))+0.03)
      
      AD, ssd, m2lnL = likelihood_maximization(A_i-D_i, err_A_i, n_pars=2)
      if m2lnL < m2lnL_min:
          m2lnL_min = m2lnL
          T_0_min = T_0
          AD_min   = AD
          ssd_min = ssd
          AD_i_min = A_i-D_i
          
  AD, ssd, T_0, AD_i = AD_min, ssd_min, T_0_min, AD_i_min

  return AD_i, AD, T_0, ssd
    
def IC(x, y, ey, order_max, verbose=True):
  
  N = float(len(x))
  e0s = np.linspace(0.0, 0.2, 201)
  orders, ICs = [], []
  for order in range(0, order_max+1):
    
      IC_min = 1.e90
      for e0 in e0s:
          
          Var = ey**2+e0**2
          pars = np.polyfit(x, y, order, w=1.0/np.sqrt(Var))[::-1]
          
          y_model = 0.0
          for i in range(0, len(pars)):  y_model = y_model + pars[i]*x**i
              
          residuals = y - y_model
          m2lnL = np.sum(np.log(Var)+residuals**2/Var)
          
          k = float(len(pars) + 1)
          penalty = k*np.log(N)
          
          IC = m2lnL + penalty
          
          if IC < IC_min:
              IC_min    = IC
              pars_min  = pars
      orders.append(order)
      ICs.append(IC_min)
     
  ICs = np.array(ICs)

  dICs = ICs - min(ICs)
      
  wICs = np.exp(-0.5*dICs)/np.sum(np.exp(-0.5*dICs))
  evidence_ratios = max(wICs)/wICs
  appropriated = []
  for evidence_ratio in evidence_ratios:
      if evidence_ratio >  13.0:  appropriated.append('no')
      if evidence_ratio <= 13.0:  appropriated.append('yes')
  
  if verbose:  
      print('model   bayesian   evidence   good')
      print('order   weight     ratio      fit?')
      print('-----   --------   --------   ----')
      for order, wIC, evidence_ratio, good in zip(orders, wICs, evidence_ratios, appropriated):
          if evidence_ratio <=99.0: print('%i       %5.3f    %6.1f       %-s' % (order, wIC, evidence_ratio, good))
          if evidence_ratio  >99.0: print('%i       %5.3f       >99       %-s' % (order, wIC, good))
      print('')

  return appropriated, pars_min
