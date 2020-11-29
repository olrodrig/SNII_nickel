import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

scale = 2.0
mpl.rc('text', usetex=True)
mpl.rc('axes', labelsize=12.0*scale)
mpl.rcParams.update({'font.size': 11*scale})

def minmax(_minmax, nd_):
  _min, _max = min(_minmax), max(_minmax)
  d_ = (_max-_min)*0.01
  _min, _max = _min-d_*nd_, _max+d_*nd_
  d_ = (_max-_min)*0.01
  return _min, _max, d_

def draw_ticks(ax):  
  x_majors = ax.xaxis.get_majorticklocs()
  x_minor  = ((max(x_majors) - min(x_majors)) / float(len(x_majors)-1))/5.0  
  ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=x_minor)) 
  
  y_majors = ax.yaxis.get_majorticklocs()
  y_minor  =  ((max(y_majors) - min(y_majors)) / float(len(y_majors)-1))/5.0  
  ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=y_minor))
  
  ax.tick_params('both', length=4, width=0.8, which='major', direction='in')
  ax.tick_params('both', length=2, width=0.8, which='minor', direction='in')
  ax.xaxis.set_ticks_position('both')
  ax.yaxis.set_ticks_position('both')

def plot_logMNi(panels, self, sn='', figure_name=''):
  
  band      = self.band
  Dt        = self.Dt
  mag       = self.mag
  emag      = self.emag
  Dt_i      = self.Dt_i
  logMNi_i  = self.logMNi_i
  elogMNi_i = self.elogMNi_i
      
  ssd       = self.ssd
  err_B     = self.err_B
    
  panels = panels.split(',')
  
  if 'fdep<1' in panels and 'fdep<1' not in logMNi_i.keys():  panels.remove('fdep<1')
  
  if len(panels) == 2 and (('lc' in panels[0] and 'lc' in panels[1]) or ('fdep' in panels[0] and 'fdep' in panels[1])):
      wspace = 0.28
      same_yaxis = True
  else:
      wspace = 0.40
      same_yaxis = False
  
  fig = plt.figure(figsize=(5.0*scale,2.5*scale))
  fig.subplots_adjust(wspace=wspace, hspace=0.12, left=0.139, bottom=0.147, right=0.975, top=0.98)
  
  ax = {}
  for i_panel in range(len(panels)):
      
      panel     = panels[i_panel]
      ax[panel] = fig.add_subplot(1,len(panels),i_panel+1)
      
      #light curves
      if 'lc' in panel:
          
          mask_in  = (Dt>=self.tmin) & (Dt<=self.tmax)
          mask_out = (Dt< self.tmin) | (Dt> self.tmax)
              
          ax[panel].plot([Dt[mask_in], Dt[mask_in]], [mag[mask_in]-emag[mask_in], mag[mask_in]+emag[mask_in]], '-', color='gray', lw=0.5*scale)
          ax[panel].plot(Dt[mask_in], mag[mask_in], 'o', color='none', mec='b', ms=6.0*scale, mew=0.7*scale)
          
          if panel == 'lc-zoom':
              x_minmax = Dt[mask_in].copy()
              y_minmax = np.array([min(mag[mask_in]-emag[mask_in]), max(mag[mask_in]+emag[mask_in])])
          if panel == 'lc':
              ax[panel].plot([Dt[mask_out], Dt[mask_out]], [mag[mask_out]-emag[mask_out], mag[mask_out]+emag[mask_out]], '-', color='gray', lw=0.5*scale)
              ax[panel].plot(Dt[mask_out], mag[mask_out], 'o', color='none', mec='gray', ms=4.0*scale, mew=0.5*scale)
              x_minmax = Dt.copy()
              y_minmax = np.array([min(mag-emag), max(mag+emag)])
              
      elif 'fdep' in panel:
          logMNi = 99
          if panel == 'fdep=1': 
              if self.fdep == True :  logMNi = self.logMNi_unc
              if self.fdep == False:  logMNi = self.logMNi
          if panel == 'fdep<1': logMNi = self.logMNi
          ax[panel].plot(Dt_i, logMNi_i[panel], 'o', color='none', mec='b', ms=6.0*scale, mew=0.7*scale)
          ax[panel].plot([Dt_i,Dt_i], [logMNi_i[panel]-elogMNi_i[panel],logMNi_i[panel]+elogMNi_i[panel]], '-', lw=0.5*scale, color='gray', zorder=1)
          ax[panel].plot([min(Dt_i),max(Dt_i)], [logMNi,logMNi], '-k', lw=1.0*scale)
          ax[panel].plot([min(Dt_i),max(Dt_i)], [logMNi-ssd[panel],logMNi-ssd[panel]], '--k', lw=1.0*scale)
          ax[panel].plot([min(Dt_i),max(Dt_i)], [logMNi+ssd[panel],logMNi+ssd[panel]], '--k', lw=1.0*scale)
          
          x_minmax = Dt_i
          y_minmax = np.array([min(logMNi_i[panel]-elogMNi_i[panel]),logMNi-err_B,max(logMNi_i[panel]+elogMNi_i[panel]),logMNi+err_B])
      
      ndx, ndy = 8.0, 8.0
      xmin, xmax, dx = minmax(x_minmax, ndx)
      ymin, ymax, dy = minmax(y_minmax, ndy)
      ax[panel].set_xlim(xmin, xmax)
      ax[panel].set_ylim(ymin, ymax)
      
      if 'lc' not in panel:
          if panel == 'fdep<1':  logMNi = self.logMNi
          if panel == 'fdep=1':
              if self.fdep == True :  logMNi = self.logMNi_unc
              if self.fdep == False:  logMNi = self.logMNi
          ax[panel].plot([max(Dt_i)+3.0*dx,max(Dt_i)+3.0*dx], [logMNi-err_B,logMNi+err_B], '-r', lw=1.0*scale)
      
      draw_ticks(ax[panel])
      
      legends = False
      if i_panel == 0:
          if sn == '':  label = '$N\!=\!'+str(len(Dt_i))+'$'
          if sn != '':  label = '$\mathrm{'+sn+'}\,(N\!=\!'+str(len(Dt_i))+')$'
          ax[panel].plot(0,0,',w', label=label)
          legends = True
      if panel == 'fdep<1':
          T_0_str = '%3i' % self.T_0
          label = '$T_0\!=\!'+T_0_str+'\,\mathrm{d}$'
          ax[panel].plot(0,0,',w', label=label)
          legends = True
      if 'lc' not in panel:
          rms_str = '%5.3f' % ssd[panel]
          label = '$\hat{\sigma}\!=\!'+rms_str+'$'
          ax[panel].plot(0,0,',w', label=label)
          legends = True
      
      if legends:
          mpl.rcParams['legend.handlelength'] = -0.2
          ax[panel].legend(loc='best', ncol=1, labelspacing=0.06, borderpad=0.2, edgecolor='none')
      
      ax[panel].set_xlabel('$\Delta t\,\mathrm{[d]}$ ')
      
      if i_panel == 0:
          if 'lc'   in panel:  ax[panel].set_ylabel('$'+band+'$-band magnitude')
          if 'fdep' in panel:  ax[panel].set_ylabel(r'$\log\left(M_{^{56}\mathrm{Ni}}/\mathrm{M}_{\odot}\right)\,\mathrm{[dex]}$')
      if i_panel == 1 and same_yaxis == False:
          if 'lc'   in panel:  ax[panel].set_ylabel('$'+band+'$-band magnitude')
          if 'fdep' in panel:  ax[panel].set_ylabel(r'$\log\left(M_{^{56}\mathrm{Ni}}/\mathrm{M}_{\odot}\right)\,\mathrm{[dex]}$')
          
      if panel in ['lc', 'lc-zoom']:
          ax[panel].invert_yaxis()

  if figure_name != '':  fig.savefig(figure_name, dpi=300)
