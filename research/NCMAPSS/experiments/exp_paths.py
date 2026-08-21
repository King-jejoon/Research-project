"""exp_paths.py — machine-independent locations for the NCMAPSS chain.
  DEMO   colleague GP modules (repo: research/NCMAPSS/deps/DEMO; override NCMAPSS_DEMO)
  PAPER  figure / document output dir (repo: research/NCMAPSS/paper; override NCMAPSS_PAPER)
  DATA   N-CMAPSS h5 files (override NCMAPSS_DATA; default = the original workstation path)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.environ.get('NCMAPSS_DEMO', os.path.normpath(os.path.join(HERE, '..', 'deps', 'DEMO')))
PAPER = os.environ.get('NCMAPSS_PAPER', os.path.normpath(os.path.join(HERE, '..', 'paper')))
DATA = os.environ.get('NCMAPSS_DATA', '/Users/a1/Desktop/project/pytorch_practice/data')
