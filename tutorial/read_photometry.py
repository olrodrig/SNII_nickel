import numpy as np

def read_photometry(input_file):
  t  = np.genfromtxt(input_file, usecols=0)
  m  = np.genfromtxt(input_file, usecols=1)
  em = np.genfromtxt(input_file, usecols=2)
  
  return t, m, em
