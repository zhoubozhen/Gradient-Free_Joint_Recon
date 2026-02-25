'''''
author  - Seonyeong Park
date    - Dec 17, 2021

FORWARD PROPAGATION USING K-WAVE (incomplete, for use of k-Wave)

'''

def forward_prop(kgrid, medium, p0, sensor, input_args, bwd_operator, iter, saving_dir):
  # Set initial pressure to be the lasted estimate of p0
  source = {'p0': p0}

  # Forward acoustic simulation
  pest = kspaceFirstOrder3DG(kgrid, medium, source, sensor, *input_args)

  return pest
