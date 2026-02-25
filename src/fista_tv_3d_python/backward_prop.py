'''''
author  - Seonyeong Park
date    - Dec 17, 2021

BACKWARD PROPAGATION USING K-WAVE (incomplete, for use of k-Wave)

'''

def backward_prop(kgrid, medium, difference, sensor, input_args, bwd_operator, iter, saving_dir):
  sensor['record'] = 'p_final'
  source = {'p_mask': sensor['mask'], \
            'p_mode': 'additive'}

  # MATLAB VERSION HERE:
    # source.p = difference(:, end:-1:1);
    # index = find(source.p_mask > 0);
    # source.p = bsxfun(@times, source.p, kgrid.dx./(unique(medium.sound_speed(index)).*(2*kgrid.dt)));

  if bwd_operator == 'adjoint':
    # ADJOINT INPUT COMPUTATION NEEDS TO BE UPDATED.
    p0est = kspaceFirstOrder3DG(kgrid, medium, source, sensor, *input_args)
  return p0est
