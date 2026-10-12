"""Registration schedules matching the WMA command-line presets."""

import copy
import math

PAIRWISE_PRESETS = {
    'affine': {
        'points_per_fiber': 10,
        'sigma': [20, 10, 10, 5, 5],
        'iterations': [3, 3, 3, 3, 3],
        'maxfun': [50, 80, 80, 80, 100],
        'mean_brain_size': [2000, 3000, 3000, 3500, 4000],
        'subject_brain_size': [1000, 1500, 1500, 1500, 1500],
        'initial_step': [10, 8, 5, 3, 2],
        'final_step': [2, 2, 1, 1, 1],
        'modes': ['Affine', 'Affine', 'Affine', 'Affine', 'Affine'],
    },
    'rigid': {
        'points_per_fiber': 10,
        'sigma': [20, 10, 10, 5, 5],
        'iterations': [3, 3, 3, 3, 3],
        'maxfun': [50, 80, 80, 80, 100],
        'mean_brain_size': [2000, 3000, 3000, 3500, 4000],
        'subject_brain_size': [1000, 1500, 1500, 1500, 1500],
        'initial_step': [10, 8, 5, 3, 2],
        'final_step': [2, 2, 1, 1, 1],
        'modes': ['Rigid', 'Rigid', 'Rigid', 'Rigid', 'Rigid'],
    },
    'affine_fast': {
        'points_per_fiber': 10,
        'sigma': [30, 10, 7.5, 5],
        'iterations': [2, 2, 2, 2],
        'maxfun': [45, 60, 75, 90],
        'mean_brain_size': [1000, 3000, 4000, 5000],
        'subject_brain_size': [250, 1500, 1750, 2000],
        'initial_step': [10, 5, 5, 5],
        'final_step': [5, 2, 2, 2],
        'modes': ['Affine', 'Affine', 'Affine', 'Affine'],
    },
    'rigid_affine_fast': {
        'points_per_fiber': 10,
        'sigma': [30, 30, 10, 7.5, 5],
        'iterations': [2, 2, 2, 2, 2],
        'maxfun': [45, 45, 60, 75, 90],
        'mean_brain_size': [1000, 1000, 3000, 4000, 5000],
        'subject_brain_size': [250, 250, 1500, 1750, 2000],
        'initial_step': [10, 10, 5, 5, 5],
        'final_step': [5, 5, 2, 2, 2],
        'modes': ['Rigid', 'Rigid', 'Rigid', 'Affine', 'Affine'],
    },
    'affine_neonate': {
        'points_per_fiber': 10,
        'sigma': [20, 10, 10, 5, 5],
        'iterations': [3, 3, 3, 3, 3],
        'maxfun': [50, 80, 80, 80, 100],
        'mean_brain_size': [4000, 4000, 4000, 4500, 6000],
        'subject_brain_size': [1500, 1750, 2000, 2000, 2500],
        'initial_step': [10, 8, 5, 3, 2],
        'final_step': [2, 2, 1, 1, 1],
        'modes': ['Affine', 'Affine', 'Affine', 'Affine', 'Affine'],
    },
    'nonrigid_neonate': {
        'points_per_fiber': 10,
        'sigma': [5, 3, 2, 1.5],
        'iterations': [3, 3, 2, 2],
        'maxfun': [205, 390, 670, 670],
        'mean_brain_size': [4000, 4500, 5000, 6000],
        'subject_brain_size': [2000, 2500, 3000, 3500],
        'initial_step': [5, 4, 3, 2],
        'final_step': [3, 3, 2, 1],
        'grid_resolution': [4, 5, 6, 6],
        'modes': ['Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid'],
    },
    'nonrigid': {
        'points_per_fiber': 3,
        'sigma': [5, 3, 2, 2, 2],
        'iterations': [10, 10, 8, 5, 2],
        'maxfun': [205, 390, 670, 1049, 1556],
        'mean_brain_size': [1000, 1000, 1000, 1000, 1000],
        'subject_brain_size': [500, 500, 500, 500, 500],
        'initial_step': [5, 4, 3, 2, 1.5],
        'final_step': [3, 3, 2, 1, 1],
        'grid_resolution': [4, 5, 6, 7, 8],
        'modes': ['Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid'],
    },
    'nonrigid_10': {
        'points_per_fiber': 3,
        'sigma': [5, 3, 2, 2, 2, 2, 2],
        'iterations': [10, 10, 8, 5, 4, 3, 2],
        'maxfun': [205, 390, 670, 1049, 1556, 2200, 3020],
        'mean_brain_size': [1000, 1000, 1000, 1000, 1000, 1000, 1000],
        'subject_brain_size': [500, 500, 500, 500, 500, 500, 500],
        'initial_step': [5, 4, 3, 2, 1.5, 1.5, 1.5],
        'final_step': [3, 3, 2, 1, 1, 1, 1],
        'grid_resolution': [4, 5, 6, 7, 8, 9, 10],
        'modes': ['Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid'],
    },
    'affineTEST': {
        'points_per_fiber': 5,
        'sigma': [30, 10, 7.5],
        'iterations': [1, 1, 1],
        'maxfun': [60, 80, 100],
        'mean_brain_size': [1500, 2000, 3000],
        'subject_brain_size': [100, 500, 1000],
        'initial_step': [5, 5, 5],
        'final_step': [2, 2, 2],
        'modes': ['Affine', 'Affine', 'Affine'],
    },
    'nonrigidTEST': {
        'points_per_fiber': 15,
        'sigma': [3, 2, 1, 1, 1, 1],
        'iterations': [1, 1, 1, 1, 1, 1],
        'maxfun': [10, 10, 10, 10, 10, 10],
        'mean_brain_size': [1000, 1000, 1000, 1000, 1000, 1000],
        'subject_brain_size': [100, 100, 100, 100, 100, 100],
        'initial_step': [5, 3, 1, 1, 1, 1],
        'final_step': [2, 1, 1, 1, 1, 1],
        'grid_resolution': [3, 4, 5, 6, 8, 10],
        'modes': ['Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid'],
    },
}

# Groupwise presets share schedules where the CLI settings agree.
GROUPWISE_PRESETS = {
    'affine': PAIRWISE_PRESETS['affine'],
    'rigid': PAIRWISE_PRESETS['rigid'],
    'affine_fast_cheap': {
        'points_per_fiber': 5,
        'sigma': [20, 10, 10, 10],
        'iterations': [3, 3, 3, 3],
        'maxfun': [50, 80, 80, 80],
        'mean_brain_size': [2000, 3000, 3000, 3000],
        'subject_brain_size': [1000, 1000, 1000, 1000],
        'initial_step': [10, 8, 5, 3],
        'final_step': [1, 1, 0.5, 0.5],
        'modes': ['Affine', 'Affine', 'Affine', 'Affine'],
    },
    'affine_neonate': PAIRWISE_PRESETS['affine_neonate'],
    'nonrigid_neonate': PAIRWISE_PRESETS['nonrigid_neonate'],
    'nonrigid': PAIRWISE_PRESETS['nonrigid'],
    'nonrigid_lowsample': {
        'points_per_fiber': 3,
        'sigma': [5, 3, 2, 2, 2],
        'iterations': [10, 10, 8, 5, 2],
        'maxfun': [205, 390, 670, 1049, 1556],
        'mean_brain_size': [800, 800, 800, 800, 800],
        'subject_brain_size': [450, 450, 450, 450, 450],
        'initial_step': [5, 4, 3, 2, 1.5],
        'final_step': [3, 3, 2, 1, 1],
        'grid_resolution': [4, 5, 6, 7, 8],
        'modes': ['Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid', 'Nonrigid'],
    },
    'affineTEST': PAIRWISE_PRESETS['affineTEST'],
    'nonrigidTEST': PAIRWISE_PRESETS['nonrigidTEST'],
}

SCALE_FIELDS = ('sigma', 'iterations', 'maxfun', 'mean_brain_size', 'subject_brain_size', 'initial_step', 'final_step')


def registration_parameters(mode='affine', params=None, groupwise=False):
    presets = GROUPWISE_PRESETS if groupwise else PAIRWISE_PRESETS
    if mode == 'custom':
        if not params:
            raise ValueError('Custom registration requires at least one scale.')
        settings = copy.deepcopy(params)
    else:
        if mode not in presets:
            raise ValueError('Unknown registration method: %s' % mode)
        settings = copy.deepcopy(presets[mode])
        if params:
            settings.update(copy.deepcopy(params))
    count = len(settings['sigma'])
    if not count:
        raise ValueError('Registration requires at least one scale.')
    if len(settings['modes']) != count:
        # Reduced schedules can override the numeric fields of a preset.
        if len(set(settings['modes'])) == 1:
            settings['modes'] = settings['modes'][:1] * count
        else:
            raise ValueError('Specify a transform type for every scale.')
    for field in SCALE_FIELDS:
        values = settings[field]
        if len(values) != count or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('%s must contain one positive value per scale.' % field)
        if field in ('iterations', 'maxfun', 'mean_brain_size', 'subject_brain_size'):
            if any(int(v) != v for v in values):
                raise ValueError('%s must contain integers.' % field)
            settings[field] = [int(v) for v in values]
    modes = settings['modes']
    if any(m not in ('Affine', 'Rigid', 'Nonrigid') for m in modes):
        raise ValueError('Transform types must be Affine, Rigid, or Nonrigid.')
    if 'Nonrigid' in modes:
        if any(m != 'Nonrigid' for m in modes):
            raise ValueError('Run affine/rigid first, then nonrigid on the registered tractography.')
        grids = settings.get('grid_resolution', [])
        if len(grids) != count or any(int(g) != g or g < 3 for g in grids):
            raise ValueError('Nonrigid grid resolutions must be integers of at least 3, one per scale.')
        settings['grid_resolution'] = [int(g) for g in grids]
    points = settings['points_per_fiber']
    if int(points) != points or points < 2:
        raise ValueError('Points per fiber must be an integer of at least 2.')
    settings['points_per_fiber'] = int(points)
    return settings


def configure_scale(register, settings, scale):
    for field in SCALE_FIELDS:
        if field != 'iterations':
            setattr(register, field, settings[field][scale])
    register.mode = settings['modes'][scale]
    if register.mode == 'Nonrigid':
        register.nonrigid_grid_resolution = settings['grid_resolution'][scale]
        register.update_nonrigid_grid()
