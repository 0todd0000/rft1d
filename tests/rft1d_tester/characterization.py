'''
Characterization ("golden value") case registry.

This module defines a large set of deterministic probes of the public rft1d
API.  Each probe is a zero-argument function that returns either a numeric
result (scalar / sequence / array) or a string.

The same registry is used by two consumers:

* ``tests/gen_characterization_golden.py`` --- records the current results to
  ``tests/data-characterization/golden.npz``
* ``tests/test_characterization.py``       --- replays the registry and asserts
  that nothing has changed

The point is to pin down existing behaviour before refactoring, so that
internal restructuring can be verified to be behaviour-preserving.

Probes must be deterministic:  any probe touching ``numpy.random`` seeds it
first.
'''

import inspect

import numpy as np

import rft1d


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

CASES = {}


def case(name):
    '''Register a zero-argument probe under *name*.'''
    def decorator(fn):
        if name in CASES:
            raise ValueError(f'duplicate characterization case name: {name!r}')
        CASES[name] = fn
        return fn
    return decorator


def _register(name, fn):
    '''Register a probe built inside a loop (closures over loop variables).'''
    if name in CASES:
        raise ValueError(f'duplicate characterization case name: {name!r}')
    CASES[name] = fn


def evaluate(name):
    '''
    Run one probe and normalize its result.

    :Returns:
        ``('num', ndarray)`` or ``('str', str)``.  Probes that raise are
        recorded as strings of the form ``"<ExcType>"`` so that error
        behaviour is pinned down too.
    '''
    try:
        value = CASES[name]()
    except Exception as exc:                      # noqa: BLE001 - intentional
        return 'str', f'RAISED {type(exc).__name__}'
    return normalize(value)


def normalize(value):
    if isinstance(value, str):
        return 'str', value
    if value is None:
        return 'str', 'None'
    a = np.asarray(value, dtype=float)
    return 'num', a


def case_names():
    return sorted(CASES)


# ---------------------------------------------------------------------------
# shared fixtures (deterministic)
# ---------------------------------------------------------------------------

def _field(seed=0, nResponses=1, nodes=101, fwhm=15.0):
    np.random.seed(seed)
    return rft1d.random.randn1d(nResponses, nodes, fwhm)


def _mask(nodes=101, i0=25, i1=55):
    b = np.array([True] * nodes)
    b[i0:i1] = False
    return b


DF = {'Z': None, 'T': 8, 'X2': 8, 'F': (2, 14), 'T2': (2, 14)}
DF2 = {'Z': (1, 1), 'T': (1, 8), 'X2': (1, 8), 'F': (2, 14), 'T2': (2, 14)}
HEIGHTS = {'Z': (0.5, 2.0, 3.1, 5.0),
           'T': (0.5, 2.0, 3.1, 5.0),
           'X2': (1.0, 5.0, 12.0, 25.0),
           'F': (1.0, 3.0, 8.0, 20.0),
           'T2': (2.0, 10.0, 25.0, 60.0)}
STATS = ('Z', 'T', 'X2', 'F', 'T2')


# ---------------------------------------------------------------------------
# public API inventory  (guards "no API changes")
# ---------------------------------------------------------------------------

def _describe_module(mod, prefix):
    lines = []
    for name in sorted(dir(mod)):
        if name.startswith('_'):
            continue
        obj = getattr(mod, name)
        if inspect.ismodule(obj):
            continue
        full = f'{prefix}.{name}'
        if inspect.isclass(obj):
            lines.append(f'{full} : class')
            for mname in sorted(dir(obj)):
                if mname.startswith('_'):
                    continue    # private members are free to change
                m = getattr(obj, mname)
                if callable(m):
                    try:
                        sig = str(inspect.signature(m))
                    except (TypeError, ValueError):
                        sig = '(?)'
                    lines.append(f'{full}.{mname}{sig}')
                elif isinstance(m, property):
                    lines.append(f'{full}.{mname} : property')
        elif callable(obj):
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = '(?)'
            lines.append(f'{full}{sig}')
        else:
            lines.append(f'{full} : {type(obj).__name__}')
    return lines


@case('api.toplevel')
def _api_toplevel():
    names = sorted(n for n in dir(rft1d) if not n.startswith('_'))
    return '\n'.join(f'{n} : {type(getattr(rft1d, n)).__name__}' for n in names)


@case('api.__all__')
def _api_all():
    return repr(rft1d.__all__)


@case('api.distributions')
def _api_distributions():
    return '\n'.join(_describe_module(rft1d.distributions, 'distributions'))


@case('api.geom')
def _api_geom():
    return '\n'.join(_describe_module(rft1d.geom, 'geom'))


@case('api.prob')
def _api_prob():
    return '\n'.join(_describe_module(rft1d.prob, 'prob'))


@case('api.random')
def _api_random():
    return '\n'.join(_describe_module(rft1d.random, 'random'))


@case('api.data')
def _api_data():
    return '\n'.join(_describe_module(rft1d.data, 'data'))


# ---------------------------------------------------------------------------
# prob.rft  and  EC densities
# ---------------------------------------------------------------------------

def _add_rft_cases():
    resels = [(1, 10.0), (1, 2.5), (2, 6.9), (1, 100.0), (3, 0.4)]
    for stat in STATS:
        df = DF2[stat]
        for z in HEIGHTS[stat]:
            for ri, R in enumerate(resels):
                for version in ('spm8', 'spm12'):
                    for n in (1, 2):
                        nm = f'prob.rft.max.{stat}.z{z}.R{ri}.{version}.n{n}'
                        _register(nm, (lambda stat=stat, z=z, R=R, df=df,
                                       version=version, n=n:
                                       rft1d.prob.rft(1, 0, stat, z, df, R,
                                                      n=n, version=version)))
        # cluster- and set-level
        for z in HEIGHTS[stat][:3]:
            for k in (0.0, 0.5, 2.0):
                for c in (1, 2, 3):
                    nm = f'prob.rft.set.{stat}.z{z}.k{k}.c{c}'
                    _register(nm, (lambda stat=stat, z=z, k=k, c=c,
                                   df=DF2[stat]:
                                   rft1d.prob.rft(c, k, stat, z, df,
                                                  (1, 10.0))))
        # Bonferroni comparison branch
        for z in HEIGHTS[stat][:3]:
            nm = f'prob.rft.bonf.{stat}.z{z}'
            _register(nm, (lambda stat=stat, z=z, df=DF2[stat]:
                           rft1d.prob.rft(1, 0, stat, z, df, (1, 50.0),
                                          n=1, Q=101.0)))
        # expectations only
        for z in HEIGHTS[stat]:
            nm = f'prob.rft.expectations.{stat}.z{z}'
            _register(nm, (lambda stat=stat, z=z, df=DF2[stat]:
                           rft1d.prob.rft(1, 0, stat, z, df, (1, 10.0),
                                          expectations_only=True)))
        # ec densities
        for z in HEIGHTS[stat]:
            nm = f'prob.ec_density.{stat}.z{z}'
            _register(nm, (lambda stat=stat, z=z, df=DF2[stat]:
                           rft1d.prob.ec_density(stat, z, df)))
        # bonferroni p values
        for z in HEIGHTS[stat]:
            nm = f'prob.p_bonferroni.{stat}.z{z}'
            _register(nm, (lambda stat=stat, z=z, df=DF2[stat]:
                           rft1d.prob.p_bonferroni(stat, z, df, 101)))
            nm = f'prob.p_bonferroni.{stat}.z{z}.n2'
            _register(nm, (lambda stat=stat, z=z, df=DF2[stat]:
                           rft1d.prob.p_bonferroni(stat, z, df, 101, n=2)))
        # inverse survival function
        for alpha in (0.01, 0.05, 0.10):
            nm = f'prob.isf.{stat}.a{alpha}'
            _register(nm, (lambda stat=stat, alpha=alpha, df=DF2[stat]:
                           rft1d.prob.isf(stat, alpha, df, (1, 10.0), 1)))
            nm = f'prob.isf.{stat}.a{alpha}.bonf'
            _register(nm, (lambda stat=stat, alpha=alpha, df=DF2[stat]:
                           rft1d.prob.isf(stat, alpha, df, (1, 50.0), 1,
                                          Q=101.0)))
    # unknown-statistic and unknown-version error behaviour
    _register('prob.ec_density.bad_stat',
              lambda: rft1d.prob.ec_density('W', 2.0, (1, 8)))
    _register('prob.rft.bad_version',
              lambda: rft1d.prob.rft(1, 0, 'Z', 2.0, None, (1, 10.0),
                                     version='spm99'))
    _register('prob.isf.bad_stat',
              lambda: rft1d.prob.isf('W', 0.05, (1, 8), (1, 10.0), 1))
    # infinitely smooth field (second resel count == 0)
    _register('prob.rft.zero_resels',
              lambda: rft1d.prob.rft(1, 0, 'Z', 2.0, None, (1, 0)))
    _register('prob.poisson_cdf.neg', lambda: rft1d.prob.poisson_cdf(1, -1.0))
    _register('prob.poisson_cdf.zero', lambda: rft1d.prob.poisson_cdf(1, 0.0))
    _register('prob.poisson_cdf.pos', lambda: rft1d.prob.poisson_cdf(2, 1.5))
    _register('prob.rft.array_height',
              lambda: rft1d.prob.rft(1, 0, 'Z', np.array(2.5), None, (1, 10.0)))


_add_rft_cases()


# ---------------------------------------------------------------------------
# prob.RFTCalculator / RFTCalculatorResels
# ---------------------------------------------------------------------------

def _calc_probe(calc, u):
    '''Probes that need only the resel counts (work on both calculators).'''
    e = calc.expected
    return [calc.sf(u),
            calc.p.upcrossing(u),
            calc.p.cluster(0.5, u),
            calc.p.set(2, 0.5, u),
            e.number_of_upcrossings(u),
            e.number_of_suprathreshold_resels(u),
            e.resels_per_upcrossing(u)]


def _calc_probe_fwhm(calc, u):
    '''Probes that convert resels to nodes, and so need ``calc.FWHM``.'''
    e = calc.expected
    return [e.number_of_suprathreshold_nodes(u),
            e.nodes_per_upcrossing(u)]


def _add_calculator_cases():
    configs = [('Z', None, 101, 10.0),
               ('T', (1, 8), 101, 15.0),
               ('X2', (1, 8), 101, 20.0),
               ('F', (2, 14), 101, 8.0),
               ('T2', (2, 14), 101, 25.0),
               ('T', (1, 8), 101, 1.5),
               ('Z', None, 51, np.inf)]
    for stat, df, nodes, fwhm in configs:
        tag = f'{stat}.q{nodes}.w{fwhm}'
        for bonf in (False, True):
            nm = f'prob.RFTCalculator.{tag}.bonf{bonf}'
            _register(nm, (lambda stat=stat, df=df, nodes=nodes, fwhm=fwhm,
                           bonf=bonf:
                           _calc_probe(rft1d.prob.RFTCalculator(
                               stat, df, nodes, fwhm, withBonf=bonf),
                               HEIGHTS[stat][1])))
            nm = f'prob.RFTCalculator.{tag}.bonf{bonf}.nodes'
            _register(nm, (lambda stat=stat, df=df, nodes=nodes, fwhm=fwhm,
                           bonf=bonf:
                           _calc_probe_fwhm(rft1d.prob.RFTCalculator(
                               stat, df, nodes, fwhm, withBonf=bonf),
                               HEIGHTS[stat][1])))
            nm = f'prob.RFTCalculator.{tag}.bonf{bonf}.isf'
            _register(nm, (lambda stat=stat, df=df, nodes=nodes, fwhm=fwhm,
                           bonf=bonf:
                           rft1d.prob.RFTCalculator(
                               stat, df, nodes, fwhm,
                               withBonf=bonf).isf([0.01, 0.05, 0.10])))
        nm = f'prob.RFTCalculator.{tag}.attrs'
        _register(nm, (lambda stat=stat, df=df, nodes=nodes, fwhm=fwhm:
                       _calc_attrs(rft1d.prob.RFTCalculator(
                           stat, df, nodes, fwhm))))
        nm = f'prob.RFTCalculator.{tag}.repr'
        _register(nm, (lambda stat=stat, df=df, nodes=nodes, fwhm=fwhm:
                       repr(rft1d.prob.RFTCalculator(stat, df, nodes, fwhm))))
        nm = f'prob.RFTCalculator.{tag}.sf_sequence'
        _register(nm, (lambda stat=stat, df=df, nodes=nodes, fwhm=fwhm:
                       rft1d.prob.RFTCalculator(
                           stat, df, nodes, fwhm).sf(list(HEIGHTS[stat]))))

    # masked (broken) fields
    for stat in STATS:
        nm = f'prob.RFTCalculator.masked.{stat}'
        _register(nm, (lambda stat=stat:
                       _calc_probe(rft1d.prob.RFTCalculator(
                           stat, DF2[stat], _mask(), 10.0),
                           HEIGHTS[stat][1])))
        nm = f'prob.RFTCalculator.masked.{stat}.nodes'
        _register(nm, (lambda stat=stat:
                       _calc_probe_fwhm(rft1d.prob.RFTCalculator(
                           stat, DF2[stat], _mask(), 10.0),
                           HEIGHTS[stat][1])))
        nm = f'prob.RFTCalculator.masked.{stat}.attrs'
        _register(nm, (lambda stat=stat:
                       _calc_attrs(rft1d.prob.RFTCalculator(
                           stat, DF2[stat], _mask(), 10.0))))

    # resel-based calculator
    for stat in STATS:
        nm = f'prob.RFTCalculatorResels.{stat}'
        _register(nm, (lambda stat=stat:
                       _calc_probe(rft1d.prob.RFTCalculatorResels(
                           stat, DF2[stat], [1, 6.667]),
                           HEIGHTS[stat][1])))
        # FWHM is None on this calculator, so node-based expectations and the
        # __repr__ (which formats FWHM) are recorded as separate cases.
        nm = f'prob.RFTCalculatorResels.{stat}.nodes'
        _register(nm, (lambda stat=stat:
                       _calc_probe_fwhm(rft1d.prob.RFTCalculatorResels(
                           stat, DF2[stat], [1, 6.667]),
                           HEIGHTS[stat][1])))
        nm = f'prob.RFTCalculatorResels.{stat}.repr'
        _register(nm, (lambda stat=stat:
                       repr(rft1d.prob.RFTCalculatorResels(
                           stat, DF2[stat], [1, 6.667]))))
        nm = f'prob.RFTCalculatorResels.{stat}.attrs'
        _register(nm, (lambda stat=stat:
                       _calc_attrs(rft1d.prob.RFTCalculatorResels(
                           stat, DF2[stat], [1, 6.667]))))
        nm = f'prob.RFTCalculatorResels.{stat}.isf'
        _register(nm, (lambda stat=stat:
                       rft1d.prob.RFTCalculatorResels(
                           stat, DF2[stat], [1, 6.667]).isf([0.01, 0.05])))
        nm = f'prob.RFTCalculatorResels.{stat}.bonf'
        _register(nm, (lambda stat=stat:
                       _calc_probe(rft1d.prob.RFTCalculatorResels(
                           stat, DF2[stat], [1, 50.0], withBonf=True,
                           nNodes=101), HEIGHTS[stat][1])))
    _register('prob.RFTCalculatorResels.bonf_without_nNodes',
              lambda: rft1d.prob.RFTCalculatorResels('Z', None, [1, 10.0],
                                                     withBonf=True))
    _register('prob.RFTCalculator.bad_nodes',
              lambda: rft1d.prob.RFTCalculator('Z', None, 'many', 10.0))
    _register('prob.RFTCalculator.bad_nodes_2d',
              lambda: rft1d.prob.RFTCalculator('Z', None,
                                               np.ones((3, 4), dtype=bool),
                                               10.0))
    # set_fwhm / set_bonf mutators
    def _mutators():
        calc = rft1d.prob.RFTCalculator('T', (1, 8), 101, 10.0)
        out = [calc.sf(2.5)]
        calc.set_fwhm(20.0)
        out.append(calc.sf(2.5))
        calc.set_bonf(True)
        out.append(calc.sf(2.5))
        out.extend(calc.resels)
        out.append(calc.Q)
        return out
    _register('prob.RFTCalculator.mutators', _mutators)


def _calc_attrs(calc):
    return [calc.nNodes,
            -1 if calc.FWHM is None else calc.FWHM,
            calc.resels[0], calc.resels[1],
            -1 if calc.Q is None else calc.Q,
            calc.n,
            float(calc.withBonf)]


_add_calculator_cases()


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------

DISTS = {'norm': ('Z', None), 't': ('T', 8), 'chi2': ('X2', 8),
         'f': ('F', (2, 14)), 'T2': ('T2', (2, 14))}


def _dist_call(dist, method, df, args_before, args_after, **kwargs):
    '''
    Call *method* on *dist*, inserting *df* between *args_before* and
    *args_after* -- and omitting it entirely for the Gaussian distribution,
    whose signatures carry no df argument.
    '''
    fn = getattr(dist, method)
    args = list(args_before)
    if df is not None:
        args.append(df)
    args.extend(args_after)
    return fn(*args, **kwargs)


def _add_distribution_cases():
    for dname, (stat, df) in DISTS.items():
        dist = getattr(rft1d, dname)
        heights = list(HEIGHTS[stat])
        for nodes_name, nodes in (('q101', 101), ('mask', _mask())):
            for fwhm in (10.0, 1.5, np.inf):
                for bonf in (False, True):
                    tag = f'{dname}.{nodes_name}.w{fwhm}.bonf{bonf}'
                    _register(f'dist.sf.{tag}',
                              (lambda dist=dist, df=df, heights=heights,
                               nodes=nodes, fwhm=fwhm, bonf=bonf:
                               _dist_call(dist, 'sf', df, [heights],
                                          [nodes, fwhm], withBonf=bonf)))
                    _register(f'dist.isf.{tag}',
                              (lambda dist=dist, df=df, nodes=nodes,
                               fwhm=fwhm, bonf=bonf:
                               _dist_call(dist, 'isf', df,
                                          [[0.01, 0.05, 0.10]],
                                          [nodes, fwhm], withBonf=bonf)))
                    _register(f'dist.p_cluster.{tag}',
                              (lambda dist=dist, df=df, heights=heights,
                               nodes=nodes, fwhm=fwhm, bonf=bonf:
                               _dist_call(dist, 'p_cluster', df,
                                          [0.5, heights[1]], [nodes, fwhm],
                                          withBonf=bonf)))
                    _register(f'dist.p_set.{tag}',
                              (lambda dist=dist, df=df, heights=heights,
                               nodes=nodes, fwhm=fwhm, bonf=bonf:
                               _dist_call(dist, 'p_set', df,
                                          [2, 0.5, heights[1]],
                                          [nodes, fwhm], withBonf=bonf)))
        # 0D methods
        _register(f'dist.sf0d.{dname}',
                  (lambda dist=dist, df=df, heights=heights:
                   dist.sf0d(heights) if df is None
                   else dist.sf0d(heights, df)))
        _register(f'dist.isf0d.{dname}',
                  (lambda dist=dist, df=df:
                   dist.isf0d([0.01, 0.05, 0.10]) if df is None
                   else dist.isf0d([0.01, 0.05, 0.10], df)))
        _register(f'dist.sf0d.{dname}.scalar',
                  (lambda dist=dist, df=df, heights=heights:
                   dist.sf0d(heights[1]) if df is None
                   else dist.sf0d(heights[1], df)))

        # resel-based variants (present on the base class only, so they take
        # the same "user-level" df as the distribution's other methods)
        rdf = df
        _register(f'dist.sf_resels.{dname}',
                  (lambda dist=dist, rdf=rdf, heights=heights:
                   dist.sf_resels(heights, rdf, (1, 10.0))))
        _register(f'dist.isf_resels.{dname}',
                  (lambda dist=dist, rdf=rdf:
                   dist.isf_resels([0.01, 0.05], rdf, (1, 10.0))))
        _register(f'dist.p_cluster_resels.{dname}',
                  (lambda dist=dist, rdf=rdf, heights=heights:
                   dist.p_cluster_resels(0.5, heights[1], rdf, (1, 10.0))))
        _register(f'dist.p_set_resels.{dname}',
                  (lambda dist=dist, rdf=rdf, heights=heights:
                   dist.p_set_resels(2, 0.5, heights[1], rdf, (1, 10.0))))
        _register(f'dist.sf_resels.{dname}.bonf',
                  (lambda dist=dist, rdf=rdf, heights=heights:
                   dist.sf_resels(heights, rdf, (1, 50.0), withBonf=True,
                                  nNodes=101)))
        # docstring injection (the DISTFLAG/DOFFLAG decorator)
        _register(f'dist.docstring.{dname}.sf',
                  lambda dist=dist: dist.sf.__doc__ or 'None')
        _register(f'dist.docstring.{dname}.isf',
                  lambda dist=dist: dist.isf.__doc__ or 'None')
        _register(f'dist.docstring.{dname}.p_cluster',
                  lambda dist=dist: dist.p_cluster.__doc__ or 'None')
        _register(f'dist.docstring.{dname}.p_set',
                  lambda dist=dist: dist.p_set.__doc__ or 'None')
        _register(f'dist.docstring.{dname}.sf0d',
                  lambda dist=dist: dist.sf0d.__doc__ or 'None')
        _register(f'dist.docstring.{dname}.isf0d',
                  lambda dist=dist: dist.isf0d.__doc__ or 'None')
        _register(f'dist.class.{dname}',
                  lambda dist=dist: type(dist).__name__)

    # values quoted in the distributions module docstring
    _register('dist.doc.rough_nobonf',
              lambda: rft1d.norm.sf(3, 101, 1.5, withBonf=False))
    _register('dist.doc.rough_bonf',
              lambda: rft1d.norm.sf(3, 101, 1.5, withBonf=True))
    _register('dist.doc.smooth_convergence',
              lambda: [rft1d.norm.sf(3, 101, w)
                       for w in (10.0, 100.0, 1000.0, 10000.0, np.inf)])


_add_distribution_cases()


# ---------------------------------------------------------------------------
# geom
# ---------------------------------------------------------------------------

BINARY_PATTERNS = {
    'empty': [0] * 12,
    'full': [1] * 12,
    'single': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'two': [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    'start': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'end': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    'both_ends': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    'alternating': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}


def _add_geom_cases():
    for pname, pattern in BINARY_PATTERNS.items():
        b = np.array(pattern, dtype=bool)
        for merge in (False, True):
            nm = f'geom.bwlabel.{pname}.merge{merge}'
            _register(nm, (lambda b=b, merge=merge:
                           _bwlabel_probe(b, merge)))

    # estimate_fwhm
    for fwhm in (5.0, 15.0, 40.0):
        nm = f'geom.estimate_fwhm.w{fwhm}'
        _register(nm, (lambda fwhm=fwhm:
                       rft1d.geom.estimate_fwhm(_field(3, 20, 101, fwhm))))
    _register('geom.estimate_fwhm.residuals',
              lambda: rft1d.geom.estimate_fwhm(
                  _field(7, 12, 51, 10.0) - _field(7, 12, 51, 10.0).mean(0)))

    # resel counts
    for eb in (False, True):
        _register(f'geom.resel_counts.unbroken.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resel_counts(np.zeros(101),
                                                         fwhm=10.0,
                                                         element_based=eb)))
        _register(f'geom.resel_counts.broken.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resel_counts(
                      _broken_mask(), fwhm=10.0, element_based=eb)))
        _register(f'geom.resel_counts.residuals.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resel_counts(
                      _field(11, 8, 101, 10.0), fwhm=10.0, element_based=eb)))
        _register(f'geom.resels2fwhm.unbroken.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resels2fwhm((1, 10.0), 101, eb)))
        _register(f'geom.resels2fwhm.broken.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resels2fwhm((2, 6.9), 71, eb)))
        _register(f'geom.resels2fwhm_masked.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resels2fwhm_masked(
                      (2, 6.9), _broken_mask(), eb)))
        _register(f'geom.resels2fieldsize.unbroken.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resels2fieldsize((1, 10.0), 10.0,
                                                             eb)))
        _register(f'geom.resels2fieldsize.broken.eb{eb}',
                  (lambda eb=eb: rft1d.geom.resels2fieldsize((2, 6.9), 10.0,
                                                             eb)))
    _register('geom.resels2nelements.unbroken',
              lambda: rft1d.geom.resels2nelements((1, 10.0), 10.0))
    _register('geom.resels2nelements.broken',
              lambda: rft1d.geom.resels2nelements((2, 6.9), 10.0))
    _register('geom.resels2nnodes.unbroken',
              lambda: rft1d.geom.resels2nnodes((1, 10.0), 10.0))
    _register('geom.resels2nnodes.broken',
              lambda: rft1d.geom.resels2nnodes((2, 6.9), 10.0))

    # cluster metrics
    thresholds = (-1.0, 0.0, 0.5, 1.5, 2.5, 4.0)
    for seed in (0, 1, 2):
        y = _field(seed, 1, 101, 15.0)
        for u in thresholds:
            for interp in (True, False):
                for wrap in (False, True):
                    tag = f's{seed}.u{u}.i{interp}.w{wrap}'
                    _register(f'geom.cluster_metrics.{tag}',
                              (lambda y=y, u=u, interp=interp, wrap=wrap:
                               _cluster_metric_probe(y, u, interp, wrap)))
                    _register(f'geom.cluster_extents_locations.{tag}',
                              (lambda y=y, u=u, interp=interp, wrap=wrap:
                               np.hstack(
                                   rft1d.geom.ClusterMetricCalculator()
                                   .cluster_extents_locations(y, u, interp,
                                                              wrap))))
                _register(f'geom.cluster_minima.s{seed}.u{u}.i{interp}',
                          (lambda y=y, u=u, interp=interp:
                           rft1d.geom.ClusterMetricCalculator()
                           .cluster_minima(y, u, interp)))
            _register(f'geom.CMCInitialized.s{seed}.u{u}',
                      (lambda y=y, u=u: _cmc_initialized_probe(y, u)))

    # a field that is entirely suprathreshold, and one that touches both ends
    y_flat = np.ones(21) * 3.0
    _register('geom.cluster_metrics.all_supra',
              lambda: _cluster_metric_probe(y_flat, 1.0, True, False))
    y_ends = np.array([3.0, 2.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 2.0,
                       3.0])
    for wrap in (False, True):
        _register(f'geom.cluster_metrics.both_ends.w{wrap}',
                  (lambda wrap=wrap:
                   _cluster_metric_probe(y_ends, 1.5, True, wrap)))
        _register(f'geom.cluster_metrics.both_ends.nointerp.w{wrap}',
                  (lambda wrap=wrap:
                   _cluster_metric_probe(y_ends, 1.5, False, wrap)))

    _register('geom.repr.ClusterMetricCalculator',
              lambda: repr(rft1d.geom.ClusterMetricCalculator()))

    # Upcrossing directly
    for wrap in (False, True):
        for interp in (True, False):
            _register(f'geom.Upcrossing.i{interp}.w{wrap}',
                      (lambda interp=interp, wrap=wrap:
                       _upcrossing_probe(y_ends, 1.5, interp, wrap)))


def _broken_mask():
    b = np.zeros(101)
    b[25:55] = 1
    return b


def _bwlabel_probe(b, merge):
    L, n = rft1d.geom.bwlabel(b, merge_wrapped=merge)
    return np.hstack([L, n])


def _cluster_metric_probe(y, u, interp, wrap):
    calc = rft1d.geom.ClusterMetricCalculator()
    out = list(calc.cluster_extents(y, u, interp, wrap))
    out.append(calc.max_cluster_extent(y, u, interp, wrap))
    out.append(calc.mean_cluster_extent(y, u, interp, wrap))
    out.append(calc.nUpcrossings(y, u))
    out.append(calc.nMaxima(y, u))
    out.append(calc.nSuprathresholdNodes(y, u))
    out.append(calc.nSuprathresholdResels(y, u, 10.0, interp))
    out.append(calc.nUpcrossingsByExtent(y, u, 5.0, interp, wrap))
    out.append(calc.total_excursion_set_extent(y, u, interp))
    return out


def _cmc_initialized_probe(y, u):
    calc = rft1d.geom.ClusterMetricCalculatorInitialized(y, u)
    extents, minima, centroids, L = calc.get_all()
    out = [calc.n, len(extents), len(minima), len(centroids)]
    out.extend(extents)
    out.extend(minima)
    for cx, cz in centroids:
        out.extend([cx, cz])
    out.append(L.sum())
    return out


def _upcrossing_probe(y, u, interp, wrap):
    b = y >= u
    L, n = rft1d.geom.bwlabel(b, merge_wrapped=wrap)
    out = []
    for i in range(n):
        up = rft1d.geom.Upcrossing(y, L == (i + 1), interp, wrap)
        yi = up.isolate()
        x0, x1 = up.endpoints(yi, u)
        out.extend([up.extent(u), up.extent_nodes(u), x0, x1, yi.size])
    return out or [0]


_add_geom_cases()


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------

def _add_random_cases():
    for fwhm in (0.0, 5.0, 15.0, np.inf):
        for pad in (False, True):
            _register(f'random.randn1d.w{fwhm}.pad{pad}',
                      (lambda fwhm=fwhm, pad=pad:
                       _seeded(lambda: rft1d.random.randn1d(5, 31, fwhm,
                                                            pad=pad))))
        _register(f'random.randn1d.single.w{fwhm}',
                  (lambda fwhm=fwhm:
                   _seeded(lambda: rft1d.random.randn1d(1, 31, fwhm))))
        _register(f'random.multirandn1d.w{fwhm}',
                  (lambda fwhm=fwhm:
                   _seeded(lambda: rft1d.random.multirandn1d(4, 21, 3, fwhm))))
    _register('random.multirandn1d.W',
              lambda: _seeded(lambda: rft1d.random.multirandn1d(
                  4, 21, 2, 10.0, W=np.array([[1.0, 0.3], [0.3, 1.0]]))))
    _register('random.randn1d.masked',
              lambda: _seeded(lambda: np.nan_to_num(
                  rft1d.random.randn1d(4, 101, 10.0), nan=-99.0)))
    _register('random.randn1d.masked_nodes',
              lambda: _seeded(lambda: np.nan_to_num(
                  rft1d.random.randn1d(4, _mask(), 10.0), nan=-99.0)))

    for fwhm in (5.0, 15.0):
        for pad in (False, True):
            _register(f'random.Generator1D.w{fwhm}.pad{pad}.attrs',
                      (lambda fwhm=fwhm, pad=pad:
                       _generator_attrs(rft1d.random.Generator1D(
                           5, 31, fwhm, pad))))
            _register(f'random.Generator1D.w{fwhm}.pad{pad}.samples',
                      (lambda fwhm=fwhm, pad=pad:
                       _seeded(lambda: np.vstack([
                           rft1d.random.Generator1D(5, 31, fwhm, pad)
                           .generate_sample() for _ in range(3)]))))
    _register('random.Generator1D.repr',
              lambda: repr(rft1d.random.Generator1D(5, 31, 10.0)))
    _register('random.GeneratorMulti1D.repr',
              lambda: repr(rft1d.random.GeneratorMulti1D(5, 31, 2, 10.0)))
    _register('random.GeneratorMulti1D.samples',
              lambda: _seeded(lambda: rft1d.random.GeneratorMulti1D(
                  4, 21, 3, 10.0).generate_sample()))
    _register('random.Generator1D.set_fwhm',
              lambda: _seeded(_regenerate_after_set_fwhm))
    _register('random.Generator1D.bad_nodes',
              lambda: rft1d.random.Generator1D(5, 'many', 10.0))
    _register('random.Generator1D.bad_nodes_2d',
              lambda: rft1d.random.Generator1D(5, np.ones((3, 4), dtype=bool),
                                               10.0))


def _seeded(fn, seed=0):
    np.random.seed(seed)
    return fn()


def _regenerate_after_set_fwhm():
    g = rft1d.random.Generator1D(3, 21, 5.0)
    y0 = g.generate_sample()
    g.set_fwhm(20.0)
    y1 = g.generate_sample()
    return np.vstack([y0, y1])


def _generator_attrs(g):
    return [g.nResponses, g.nNodes, g.FWHM, g.SD,
            -1 if g.SCALE is None else g.SCALE,
            -1 if g.q is None else g.q,
            -1 if g.i0 is None else g.i0,
            -1 if g.i1 is None else g.i1,
            float(g.pad)]


_add_random_cases()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@case('data.weather.shapes')
def _weather_shapes():
    d = rft1d.data.weather()
    return [len(d)] + [v for k in sorted(d) for v in d[k].shape]


@case('data.weather.checksums')
def _weather_checksums():
    d = rft1d.data.weather()
    return [d[k].sum() for k in sorted(d)]


@case('data.weather.keys')
def _weather_keys():
    return repr(sorted(rft1d.data.weather()))
