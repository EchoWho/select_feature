import itertools
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import scipy
import time

N_PROCESSES = 8

try:
    import os
    import sys

    import cplex
    def opt_raw(b, C):
        n_vars = b.shape[0]

        p = cplex.Cplex()
        p.set_results_stream(None)
        p.set_log_stream(None)

        p.objective.set_sense(p.objective.sense.maximize)

        obj = 2 * b
        p.variables.add(obj = obj.flatten(),
                        lb = [-cplex.infinity]*n_vars, ub = [cplex.infinity]*n_vars)

        Q = -2 * C
        inds = np.arange(n_vars)
        p.objective.set_quadratic([(inds, v.flatten()) for v in Q])

        try:
            p.solve()
        except cplex.exceptions.CplexSolverError:
            raise ValueError('CPLEX Optimizer failed')

        return np.array(p.solution.get_values())


    def opt_raw_l1_constraint(b, C, l1_eps):
        n_vars = b.shape[0]

        p = cplex.Cplex()
        p.set_results_stream(None)
        p.set_log_stream(None)

        p.objective.set_sense(p.objective.sense.maximize)

        obj = np.hstack((2 * b, np.zeros(b.shape)))
        p.variables.add(obj = obj, lb = [-l1_eps]*n_vars + [0]*n_vars, ub = [l1_eps]*(n_vars*2))

        Q = -2 * C
        inds = np.arange(n_vars)
        p.objective.set_quadratic([(inds, v) for v in Q] + [cplex.SparsePair()]*n_vars)

        lin = [cplex.SparsePair([i, i+n_vars], [1,-1]) for i in range(n_vars)]
        lin += [cplex.SparsePair([i, i+n_vars], [-1,-1]) for i in range(n_vars)]
        lin += [cplex.SparsePair(np.arange(n_vars, 2*n_vars), np.ones(n_vars))]
        rhs = [0]*n_vars + [0]*n_vars + [l1_eps]
        p.linear_constraints.add(lin_expr=lin, rhs=rhs, senses='L'*(2*n_vars+1))

        try:
            p.solve()
        except cplex.exceptions.CplexSolverError:
            raise ValueError('CPLEX Optimizer failed')

        return np.array(p.solution.get_values()[:n_vars])


    def opt_raw_l1_regularized(b, C, l1_lam):
        n_vars = b.shape[0]

        p = cplex.Cplex()
        p.set_results_stream(None)
        p.set_log_stream(None)

        p.objective.set_sense(p.objective.sense.maximize)

        obj = np.hstack((2 * b, l1_lam * np.ones(b.shape)))
        p.variables.add(obj = obj, lb = [-cplex.infinity]*n_vars + [0]*n_vars, ub = [cplex.infinity]*(n_vars*2))

        Q = -2 * C
        inds = np.arange(n_vars)
        p.objective.set_quadratic([(inds, v) for v in Q] + [cplex.SparsePair()]*n_vars)

        lin = [cplex.SparsePair([i, i+n_vars], [1,-1]) for i in range(n_vars)]
        lin += [cplex.SparsePair([i, i+n_vars], [-1,-1]) for i in range(n_vars)]
        rhs = [0]*n_vars + [0]*n_vars
        p.linear_constraints.add(lin_expr=lin, rhs=rhs, senses='L'*(2*n_vars))

        try:
            p.solve()
        except cplex.exceptions.CplexSolverError:
            raise ValueError('CPLEX Optimizer failed')

        return np.array(p.solution.get_values()[:n_vars])


    def opt_raw_l2_constraint(b, C, l2_eps):
        n_vars = b.shape[0]

        p = cplex.Cplex()
        p.set_results_stream(None)
        p.set_log_stream(None)

        p.objective.set_sense(p.objective.sense.maximize)

        obj = 2 * b
        p.variables.add(obj = obj, lb = [-l2_eps]*n_vars, ub = [l2_eps]*n_vars)

        Q = -2 * C
        inds = np.arange(n_vars)
        p.objective.set_quadratic([(inds, v) for v in Q])

        p.quadratic_constraints.add(rhs = l2_eps*l2_eps, quad_expr = (inds, inds, np.ones(n_vars)), sense='L')

        try:
            p.solve()
        except cplex.exceptions.CplexSolverError:
            raise ValueError('CPLEX Optimizer failed')

        return np.array(p.solution.get_values())


    def opt_raw_l2_regularized(b, C, l2_lam):
        n_vars = b.shape[0]

        p = cplex.Cplex()
        p.set_results_stream(None)
        p.set_log_stream(None)

        p.objective.set_sense(p.objective.sense.maximize)

        obj = 2 * b
        p.variables.add(obj = obj, lb = [-cplex.infinity]*n_vars, ub = [cplex.infinity]*n_vars)

        Q = -2 * (C + l2_lam * np.identity(C.shape[0]))
        inds = np.arange(n_vars)
        p.objective.set_quadratic([(inds, v) for v in Q])

        try:
            p.solve()
        except cplex.exceptions.CplexSolverError:
            raise ValueError('CPLEX Optimizer failed')

        return np.array(p.solution.get_values())


    def opt_raw_combined(b, C, l1_lam=None, l2_lam=None, l1_eps=None, l2_eps=None, l1_costs=None):
        n_vars = b.shape[0]
        inds = np.arange(n_vars)

        l1 = l1_lam is not None or l1_eps is not None
        if l1_costs is None:
            l1_costs = np.ones(n_vars)

        p = cplex.Cplex()
        p.set_results_stream(None)
        p.set_log_stream(None)

        p.objective.set_sense(p.objective.sense.maximize)

        # Variable bounds
        if l1_eps is not None or l2_eps is not None:
            if l1_eps is None:
                eps = l2_eps
            elif l2_eps is None:
                eps = l1_eps
            else:
                eps = np.min([l1_eps, l2_eps])

            lb = [-eps]*n_vars
            ub = [eps]*n_vars
        else:
            lb = [-cplex.infinity]*n_vars
            ub = [cplex.infinity]*n_vars

        # Linear objective
        if l1:
            if l1_lam is not None:
                obj = np.hstack((2 * b, l1_lam * l1_costs))
                p.variables.add(obj = obj, lb = lb + [0]*n_vars,
                                ub = ub + [cplex.infinity]*n_vars)
            else:
                obj = np.hstack((2 * b, np.zeros(b.shape)))
                p.variables.add(obj = obj, lb = lb + [0]*n_vars,
                                ub = ub + [l1_eps]*n_vars)
        else:
            obj = 2 * b
            p.variables.add(obj = obj, lb = lb,
                            ub = ub)

        # Quadratic objective
        if l2_lam is not None:
            Q = -2 * (C + l2_lam * np.identity(C.shape[0]))
        else:
            Q = -2 * C

        if l1:
            p.objective.set_quadratic([(inds, v) for v in Q] + [cplex.SparsePair()]*n_vars)
        else:
            p.objective.set_quadratic([(inds, v) for v in Q])

        # Linear constraint
        if l1:
            lin = [cplex.SparsePair([i, i+n_vars], [1,-1]) for i in range(n_vars)]
            lin += [cplex.SparsePair([i, i+n_vars], [-1,-1]) for i in range(n_vars)]
            rhs = [0]*n_vars + [0]*n_vars
            if l1_eps is not None:
                lin += [cplex.SparsePair(np.arange(n_vars, 2*n_vars), l1_costs)]
                rhs += [l1_eps]
                p.linear_constraints.add(lin_expr=lin, rhs=rhs, senses='L'*(2*n_vars+1))
            else:
                p.linear_constraints.add(lin_expr=lin, rhs=rhs, senses='L'*(2*n_vars))

        # Quadratic constraint
        if l2_eps is not None:
            p.quadratic_constraints.add(rhs = l2_eps*l2_eps, quad_expr = (inds, inds, np.ones(n_vars)), sense='L')

        try:
            p.solve()
            return np.array(p.solution.get_values()[:n_vars])
        except cplex.exceptions.CplexSolverError:
            print 'WARNING: CPLEX optimizer failed'
            return np.zeros(n_vars)

    def opt_raw_no_cplex(b, C, **args):
        return np.linalg.inv(C).dot(b)
        # return np.linalg.inv(C).dot(b)

    def opt(X, Y, eps, l1=False):
        b = np.dot(Y, X)
        C = np.dot(X.T, X)
        if l1:
            return opt_raw_l1_constraint(b, C, eps)
        else:
            return opt_raw_l2_constraint(b, C, eps)

except:
    def opt_raw_no_cplex(b, C, **args):
        return np.linalg.inv(C).dot(b)
        # return np.linalg.inv(C).dot(b)

    print 'ERROR: Couldn\'t find CPLEX'

def loss(w, X, Y):
    """Squared loss for given w, X, Y"""

    residual = Y - X.dot(w)
    return np.dot(residual, residual) / X.shape[0]

def rsquared(w, X, Y):
    """rsquared value for given w, X, Y"""

    return np.dot(Y, Y) / X.shape[0] - loss(w, X, Y)

def rsquared_bC(w, b, C):
    """rsquared value for given w, b, C"""

    return (2 * b.dot(w) - w.T.dot(C).dot(w))

def rsquared_l2_regularized(w, X, Y, lam):
    """rsquared value for given w, X, Y"""

    return np.dot(Y, Y) / X.shape[0] - loss(w, X, Y) - lam * np.dot(w, w)

def rsquared_l2_regularized_bC(w, b, C):
    """rsquared value for given w, b, C"""

    return (2 * b.dot(w) - w.T.dot(C).dot(w)) - lam * np.dot(w,w)

def rsquared_combined(w, X, Y, l1_lam=0.0, l2_lam=0.0, **kwargs):
    """rsquared value for given w, X, Y"""

    return np.dot(Y, Y) / X.shape[0] - loss(w, X, Y) - l1_lam * np.sum(np.abs(w)) - l2_lam * np.dot(w, w)

def rsquared_combined_bC(w, b, C, l1_lam=0.0, l2_lam=0.0, **kwargs):
    """rsquared value for given w, X, Y"""

    return (2 * b.dot(w) - w.T.dot(C).dot(w)) - l1_lam * np.sum(np.abs(w)) - l2_lam * np.dot(w, w)

def gradient(w, X, Y):
    residual = Y - X.dot(w)
    return residual

def gradient_bC(w, b, C):
    pass

class OptProblem(object):
    def __init__(self, X, Y, opt_func, score_func, gradient_func, args={}):
        if len(Y.shape) != 1:
            raise ValueError('Y must be 1D')

        if len(X.shape) != 2:
            raise ValueError('X must be 2D')

        #self.X = X
        #self.Y = Y

        self.b = X.T.dot(Y) / X.shape[0]
        self.C = X.T.dot(X) / X.shape[0]
        # force symmetric C
        self.C = (self.C + self.C.T) / 2.0

        self.opt_func = opt_func
        self.score_func = score_func
        self.gradient_func = gradient_func

        self.args = args

        self.lasso_threshold = 1e-4
        self.lasso_step = 1e-2
        self.lasso_tolerance = 1e-4

    def n_features(self):
        return self.b.shape[0]

    def opt(self, selected):
        bS = self.b[selected]
        CS = self.C[selected[:,np.newaxis], selected]

        return self.opt_func(bS, CS, **self.args)

    def score(self, selected, w):
        bS = self.b[selected]
        CS = self.C[selected[:,np.newaxis], selected]
        return self.score_func(w, bS, CS, **self.args)

    def omp_select_bC(self, selected, mask, w, costs):
        b_res = self.b - self.C[:, selected].dot(w)

        best_ip = 0.0
        best_k = -1

        # print 'running omp selection with %s already selected' % selected

        for k, bk in enumerate(b_res):
            if mask[k]:
                continue

            ip = bk * bk / costs[k]
            # print ip, k
            if ip > best_ip:
                best_ip = ip
                best_k = k

        return best_k

    def omp_select_groups_bC(self, selected, mask, w, costs, groups, noinv=False):
        selected_feats = [np.array([], dtype=np.int)]
        selected_feats += [np.nonzero(groups == g)[0] for g in selected]
        selected_feats = np.hstack(selected_feats)

        b_res = self.b - self.C[:, selected_feats].dot(w)

        best_ip = 0.0
        best_g = -1

        # print 'running omp selection with %s already selected' % selected

        for g in np.unique(groups):
            if mask[g]:
                continue

            sel_g = np.nonzero(groups == g)[0]
            bG = b_res[sel_g]
            CG = self.C[sel_g[:,np.newaxis], sel_g]

            if noinv:
                ip = bG.dot(bG) / costs[g]
            else:
                ip = bG.dot(np.linalg.inv(CG)).dot(bG) / costs[g]

            # print 'group %d ip %f' % (g, ip)
            # print ip, k

            if ip > best_ip:
                best_ip = ip
                best_g = g

        # print 'best was %d ip %f' % (best_g, best_ip)
        return best_g

    # def omp_select(self, selected, mask, w, costs):
    #     g = self.gradient_func(w, self.X[:, selected], self.Y)

    #     best_ip = 0.0
    #     best_k = -1

    #     # print 'running omp selection with %s already selected' % selected

    #     for k, xk in enumerate(self.X.T):
    #         if mask[k]:
    #             continue

    #         dk = xk.dot(g)
    #         ip = dk * dk / costs[k]
    #         # print ip, k
    #         if ip > best_ip:
    #             best_ip = ip
    #             best_k = k

    #     # print 'best was %d with %f' % (best_k, best_ip)

    #     return best_k

    def lasso_check(self, selected, eps, costs=None):
        new_args = self.args.copy()
        new_args['l1_eps'] = eps
        new_args['l1_costs'] = costs

        w = self.opt_func(self.b, self.C, **new_args)
        w[np.abs(w) < self.lasso_threshold] = 0.0
        new_selected = np.nonzero(w)[0]

        return not np.array_equal(new_selected, selected)

    def lasso_select(self, selected, last, costs=None):
        deps = self.lasso_step

        while not self.lasso_check(selected, last + deps, costs):
            deps *= 2

        hi = deps
        lo = 0.0
        while (hi - lo) > self.lasso_tolerance:
            up = self.lasso_check(selected, last + deps, costs)

            if up:
                down = self.lasso_check(selected, last + deps - self.lasso_tolerance, costs)
                if not down:
                    break
                else:
                    hi = deps
                    deps = (hi + lo) / 2.0
            else:
                lo = deps
                deps = (hi + lo) / 2.0

        new_args = self.args.copy()
        new_args['l1_eps'] = last + deps
        new_args['l1_costs'] = costs

        lasso_w = self.opt_func(self.b, self.C, **new_args)
        lasso_w[np.abs(lasso_w) < self.lasso_threshold] = 0.0

        lasso_score = self.score_func(lasso_w, self.b, self.C)
        lasso_selected = np.nonzero(lasso_w)[0]
        return lasso_selected, last + deps, lasso_score, lasso_w


def alg_optimal(problem, K=None, costs=None):
    n_features = problem.n_features()
    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    selected = np.zeros(0, dtype=np.int)
    sequence = [(0, 0.0, selected, problem.score(selected, np.zeros(0)))]

    for k in range(K):
        print 'Optimal on iteration %d' % k

        best_v = 0
        best_set = np.array([], dtype=np.int)
        best_w = np.array([])
        best_c = 0.0

        # Loop over all possible sets of k indices
        possible = np.arange(n_features)
        for s in itertools.combinations(possible, k+1):
            selected = np.array(s, dtype=np.int)

            w = problem.opt(selected)
            v = problem.score(selected, w)
            c = np.sum(costs[selected])

            if v > best_v:
                best_v = v
                best_w = w
                best_set = selected
                best_c = c

        sequence.append((best_v, best_c, best_set, best_w))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('selected', object), ('w', object)])


def alg_forward(problem, K=None, costs=None):
    n_features = problem.n_features()
    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    mask = np.zeros(n_features, np.bool)
    selected = np.zeros(K, np.int)
    last_score = 0.0
    sequence = [(0, last_score, -1, selected[:0], problem.score(selected[:0], np.zeros(0)), 0)]

    t0 = time.time()
    for k in range(K):
        print 'FR Iteration %d' % k

        best_gain = 0
        best_feature = -1
        best_set = np.array([], dtype=np.int)
        best_w = np.array([])

        sel = selected[:k+1]
        for f in range(n_features):
            # If feature is already selected just skip it
            if mask[f]:
                continue

            sel[k] = f

            w = problem.opt(sel)
            gain = (problem.score(sel, w) - last_score) / costs[f]

            if (gain > best_gain):
                best_gain = gain
                best_w = w
                best_feature = f

        if best_feature == -1:
            print 'Exited with no feature selected on iteration %d' % (k+1)
            break

        best_set = selected[:k+1]
        best_set[k] = best_feature
        mask[best_feature] = True

        last_score = problem.score(best_set, best_w)
        c = np.sum(costs[best_set])

        timestamp = time.time() - t0
        sequence.append((last_score, c, best_feature, best_set, best_w, timestamp))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('feature', np.int), ('selected', object), ('w', object), ('time', np.float)])


def alg_omp(problem, K=None, costs=None):
    n_features = problem.n_features()
    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    mask = np.zeros(n_features, np.bool)
    selected = np.zeros(K, np.int)
    sequence = [(0, 0.0, -1, selected[:0], problem.score(selected[:0], np.zeros(0)), 0)]

    last_w = np.zeros(0)

    t0 = time.time()
    for k in range(K):
        #print 'OMP Iteration %d' % k

        # Select feature with biggest inner product
        f = problem.omp_select_bC(selected[:k], mask, last_w, costs)

        if f == -1:
            print 'Exited with no feature selected on iteration %d' % (k+1)
            break

        selected[k] = f
        mask[f] = True

        sel = selected[:k+1]
        w = problem.opt(sel)
        v = problem.score(sel, w)
        c = np.sum(costs[sel])

        last_w = w

        timestamp = time.time() - t0
        sequence.append((v, c, f, sel, w, timestamp))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('feature', np.int), ('selected', object), ('w', object), ('time', np.float)])


def alg_lasso(problem, K=None, costs=None):
    n_features = problem.n_features()
    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    selected = np.array([], np.int)
    sequence = [(0, 0.0, selected, problem.score(selected, np.zeros(0)), 0.0, 0.0, np.zeros(n_features), 0)]

    last_l1 = 0.0

    t0 = time.time()
    while selected.shape[0] < K:
        print 'L1 Iteration %d' % selected.shape[0]

        # Select feature with biggest inner product
        selected, last_l1, lasso_score, lasso_w = problem.lasso_select(selected, last_l1, costs)

        # Refit without l1 constraint
        w = problem.opt(selected)
        v = problem.score(selected, w)
        c = np.sum(costs[selected])

        timestamp = time.time() - t0
        sequence.append((v, c, selected, w, last_l1, lasso_score, lasso_w, timestamp))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('selected', object), ('w', object),
                                       ('l1_eps', np.float), ('lasso_score', np.float), ('lasso_w', object), ('timestamp', np.float)])

def alg_lars(problem, lars_output, K=None, costs=None):
    n_features = problem.n_features()
    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    mask = np.zeros(n_features, np.bool)
    selected = np.array([], np.int)
    sequence = [(0, 0.0, selected, problem.score(selected, np.zeros(0)))]

    for line in open(lars_output):
        for v in line.split():
            v = int(v)
            if v < 0:
                print '%d removed' % (-v),
                mask[(-v)-1] = False
            else:
                print '%d added' % (v),
                mask[v-1] = True

        selected = np.nonzero(mask)[0]
        print '%d / %d features total' % (selected.shape[0], n_features)

        # Refit without l1 constraint
        w = problem.opt(selected)
        v = problem.score(selected, w)
        c = np.sum(costs[selected])

        sequence.append((v, c, selected, w))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('selected', object), ('w', object)])


def delta_selected(results):
    selected = results['selected']

    added = []
    removed = []

    for sp, sn in zip(selected[:-1], selected[1:]):
        added.append(np.setdiff1d(sn, sp))
        removed.append(np.setdiff1d(sp, sn))

    return added, removed


def recompute_costs(results, costs):
    new_results = results.copy()

    for i, sel in enumerate(new_results['selected']):
        new_results[i]['cost'] = np.sum(costs[sel])

    return new_results


def best_budgets(results):
    S = np.argsort(results['cost'])

    last = S[0]
    indices = []

    scores = results['score']
    costs = results['cost']

    for i in range(1, S.shape[0]):
        s = S[i]
        if scores[s] > scores[last]:
            if costs[s] > costs[last]:
                indices.append(last)
            last = s

    indices.append(last)
    return results[indices]

def fix_single_result(result, problem):
    for r in result:
        selected = r['selected']
        if selected.shape[0] > 0:
            w = problem.opt(selected)
            v = problem.score(selected, w)
            r['w'] = w
            r['score'] = v

    return result

def fix_results_p(results, problem, K=None, costs=None, l1=None):
    new_results = {}

    n_features = problem.n_features()
    if costs is None:
        costs = np.ones(n_features)

    for name in results:
        new_results[name] = fix_single_result(results[name], problem)

    if l1:
        t0 = time.time()
        new_results['L1'] = best_budgets(alg_lasso(problem, K, costs))
        print ('Lasso total time %f' % (time.time() - t0))

    return new_results

def all_results_p(problem, K=None, costs=None, optimal=None, l1=None):
    n_features = problem.n_features()
    if costs is None:
        costs = np.ones(n_features)

    results = []
    names = []

    t0 = time.time()
    results.append(alg_forward(problem, K, costs))
    names.append('FR')
    print ('Forward Reg. total time %f' % (time.time() - t0))

    t0 = time.time()
    results.append(alg_omp(problem, K, costs))
    names.append('OMP')
    print ('OMP total time %f' % (time.time() - t0))

    if type(optimal) is np.ndarray:
        results.append(best_budgets(recompute_costs(optimal, costs)))
        names.append('OPT')
    elif optimal:
        results.append(best_budgets(alg_optimal(problem, K, costs)))
        names.append('OPT')

    if l1:
        t0 = time.time()
        results.append(best_budgets(alg_lasso(problem, K, costs)))
        names.append('L1')
        print ('Lasso total time %f' % (time.time() - t0))

    return dict(zip(names, results))

def all_results_groups_p(problem, K=None, costs=None, groups=None, optimal=None):
    results = []
    names = []

    do_FR = 0
    if do_FR :
      t0 = time.time()
      results.append(alg_forward_groups(problem, K, costs, groups, method='group'))
      names.append('FR')
      print ('Forward Reg. total time %f' % (time.time() - t0))

      t0 = time.time()
      results.append(alg_forward_groups(problem, K, costs, groups, method='single'))
      names.append('FR SINGLE')
      print ('Forward Reg. on features total time %f' % (time.time() - t0))

    t0 = time.time()
    results.append(alg_omp_groups(problem, K, costs, groups, method='group'))
    names.append('OMP')
    print ('OMP total time %f' % (time.time() - t0))

    t0 = time.time()
    results.append(alg_omp_groups(problem, K, costs, groups, method='single'))
    names.append('OMP SINGLE')
    print ('OMP total time %f' % (time.time() - t0))

    t0 = time.time()
    results.append(alg_omp_groups(problem, K, costs, groups, method='noinv'))
    names.append('OMP NOINV')
    print ('OMP total time %f' % (time.time() - t0))

    if type(optimal) is np.ndarray:
        results.append(best_budgets(recompute_costs(optimal, costs)))
        names.append('OPT')
    elif optimal:
        results.append(best_budgets(alg_optimal_groups(problem, K, costs, groups)))
        names.append('OPT')

    return dict(zip(names, results))

def all_results(X, Y, K=None, costs=None, optimal=None, **opt_args):
    problem = OptProblem(X, Y, opt_raw_combined, rsquared_combined_bC, gradient_bC, args=opt_args)

    return all_results_p(problem, K, costs, optimal)

def all_results_no_cplex(X, Y, K=None, costs=None, groups=None, optimal=None, **opt_args):
    problem = OptProblem(X, Y, opt_raw_no_cplex, rsquared_combined_bC, gradient_bC, args=opt_args)
    if 'l2_lam' in opt_args:
        problem.C = problem.C + np.eye(problem.C.shape[0]) * opt_args['l2_lam']
        del opt_args['l2_lam']

    if groups is None:
        return all_results_p(problem, K, costs, optimal, l1=False)
    else:
        return all_results_groups_p(problem, K, costs, groups, optimal)

def all_results_bC(b, C, K=None, costs=None, groups=None, optimal=None, **opt_args):
    X = np.ones((2,b.shape[0]))
    Y = np.ones((2))

    problem = OptProblem(X, Y, opt_raw_no_cplex, rsquared_combined_bC, gradient_bC, args=opt_args)
    problem.b = b
    problem.C = C

    if groups is None:
        return all_results_p(problem, K, costs, optimal)
    else:
        return all_results_groups_p(problem, K, costs, groups, optimal)

def plot(results, names, **plot_args):
    colors = ['r', 'g', 'b', 'k']
    for res, c in zip(results, colors):
        plt.plot(res['cost'], res['score'], color=c, **plot_args)

    plt.legend(names, loc='lower right')

def averages(results, score='score', fix=True):
    costs = [r['cost'] for r in results]
    scores = [r[score] for r in results]

    combined = np.unique(np.sort(np.hstack(costs)))
    if fix:
        fixed = [fix(c,s) for (c,s) in zip(costs, scores)]
    else:
        fixed = zip(costs, scores)
    interped = [np.interp(combined, c, s) for (c,s) in fixed]

    if fix:
        return fix(combined, np.mean(interped, axis=0))
    else:
        return (combined, np.mean(interped, axis=0))

def fix(cost, score):
    eps = 1e-4
    c = np.repeat(cost, 2)[1:]
    c[1::2] = c[1::2] - eps
    return c, np.repeat(score, 2)[:-1]

def random_problem(d):
    mu = np.zeros(d)
    sig = np.ones((d,d)) * 0.6
    np.fill_diagonal(sig, 1)

    X = np.random.multivariate_normal(mu, sig, 100)
    w = np.random.uniform(0, 10, d)
    Y = X.dot(w) + np.random.normal(0, 0.1, 100)

    X = X - X.mean(axis=0)
    X = X / X.std(axis=0)

    Y = Y - Y.mean(axis=0)
    Y = Y / Y.std(axis=0)

    return X, Y

def random_group_problem(d, k):
    d = d - (d % k)

    mu = np.zeros(d)
    sig = np.ones((d,d)) * 0.6
    np.fill_diagonal(sig, 1)

    X = np.random.multivariate_normal(mu, sig, 100)
    groups = np.random.permutation(np.repeat(np.arange(k), d / k))

    w = np.random.uniform(0, 10, d)
    for g in range(k/2): w[groups == g] = 0

    Y = X.dot(w) + np.random.normal(0, 0.1, 100)

    X = X - X.mean(axis=0)
    X = X / X.std(axis=0)

    Y = Y - Y.mean(axis=0)
    Y = Y / Y.std(axis=0)

    return X, Y, groups

def alg_optimal_groups(problem, K=None, costs=None, groups=None):
    n_features = problem.n_features()

    if groups is None:
        groups = np.arange(n_features)
    n_groups = np.unique(groups).shape[0]

    if K is None:
        K = n_groups

    if costs is None:
        costs = np.ones(n_groups)

    selected = np.zeros(0, dtype=np.int)
    sequence = [(0, 0.0, selected, selected, problem.score(selected, np.zeros(0)))]

    for k in range(K):
        print 'Optimal on iteration %d' % k

        best_v = 0
        best_set = np.array([], dtype=np.int)
        best_groups = np.array([], dtype=np.int)
        best_w = np.array([])
        best_c = 0.0

        # Loop over all possible sets of k indices
        possible = np.unique(groups)
        for s in itertools.combinations(possible, k+1):
            selected_groups = np.array(s, dtype=np.int)
            selected = [np.nonzero(groups == g)[0] for g in s]
            selected = np.hstack(selected)

            w = problem.opt(selected)
            v = problem.score(selected, w)
            c = np.sum(costs[selected_groups])

            if v > best_v:
                best_v = v
                best_w = w
                best_set = selected
                best_groups = selected_groups
                best_c = c

        sequence.append((best_v, best_c, best_set, best_groups, best_w))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('selected', object),
                                       ('selected_groups', object), ('w', object)])

def alg_forward_groups(problem, K=None, costs=None, groups=None, method='group'):
    n_features = problem.n_features()

    if groups is None:
        groups = np.arange(n_features)
    n_groups = np.max(groups) + 1

    if K is None:
        K = n_groups

    if costs is None:
        costs = np.ones(n_groups)

    mask = np.zeros(n_groups, np.bool)
    selected_groups = np.zeros(K, np.int)
    last_score = 0.0
    sequence = [(0, last_score, -1, selected_groups[:0], selected_groups[:0],
                 problem.score(selected_groups[:0], np.zeros(0)), 0)]

    t0 = time.time()
    for k in range(K):
        print 'FR Iteration %d' % k

        best_gain = 0
        best_group = -1

        if method == 'group':
            sel = selected_groups[:k+1]
            for g in range(n_groups):
                # If feature is already selected just skip it
                if mask[g]:
                    continue

                sel[k] = g

                sel_feats = [np.nonzero(groups == g)[0] for g in sel]
                sel_feats = np.hstack(sel_feats)

                w = problem.opt(sel_feats)
                gain = (problem.score(sel_feats, w) - last_score) / costs[g]

                if (gain > best_gain):
                    best_gain = gain
                    best_group = sel[k]
        else:
            sel = selected_groups[:k]
            sel_feats = [np.array([], dtype=np.int)] + [np.nonzero(groups == g)[0] for g in sel]
            sel_feats = np.hstack(sel_feats)
            sel_feats.resize(sel_feats.shape[0] + 1)

            for f in range(n_features):
                # If feature is already selected just skip it
                if mask[groups[f]]:
                    continue

                sel_feats[-1] = f

                w = problem.opt(sel_feats)
                gain = (problem.score(sel_feats, w) - last_score) / costs[groups[f]]

                if (gain > best_gain):
                    best_gain = gain
                    best_group = groups[f]

        if best_group == -1:
            print 'Exited with no group selected on iteration %d' % (k+1)
            break

        mask[best_group] = True
        best_groups = selected_groups[:k+1]
        best_groups[k] = best_group

        best_set = [np.nonzero(groups == g)[0] for g in best_groups]
        best_set = np.hstack(best_set)

        best_w = problem.opt(best_set)

        last_score = problem.score(best_set, best_w)
        c = np.sum(costs[best_groups])

        timestamp = time.time() - t0
        sequence.append((last_score, c, best_group, best_set, best_groups, best_w, timestamp))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                                       ('selected', object), ('selected_groups', object),
                                       ('w', object), ('time', np.float)])


def alg_omp_groups(problem, K=None, costs=None, groups=None, method='group'):
    n_features = problem.n_features()

    if groups is None:
        groups = np.arange(n_features)
    n_groups = np.max(groups) + 1

    if K is None:
        K = n_groups

    if costs is None:
        costs = np.ones(n_groups)

    mask = np.zeros(n_groups, np.bool)
    selected_groups = np.zeros(K, np.int)
    sequence = [(0, 0.0, -1, selected_groups[:0], selected_groups[:0],
                 problem.score(selected_groups[:0], np.zeros(0)), 0)]

    last_w = np.zeros(0)

    t0 = time.time()
    for k in range(K):
        #print 'OMP Iteration %d' % k

        if method == 'single':
            mask_feats = np.zeros(n_features, np.bool)
            for g in selected_groups[:k]:
                mask_feats[groups == g] = True
            selected_feats = np.nonzero(mask_feats)[0]

            # Select feature with biggest inner product
            f = problem.omp_select_bC(selected_feats, mask_feats, last_w,
                                      np.array([costs[groups[f]] for f in range(n_features)]))
            if f == -1:
                print 'Exited with no feature selected on iteration %d' % (k+1)
                break

            g = groups[f]
        else:
            # Select feature with biggest inner product
            if method == 'noinv':
                g = problem.omp_select_groups_bC(selected_groups[:k], mask, last_w, costs, groups, noinv=True)
            else:
                g = problem.omp_select_groups_bC(selected_groups[:k], mask, last_w, costs, groups)
            if g == -1:
                print 'Exited with no group selected on iteration %d' % (k+1)
                break

        selected_groups[k] = g
        mask[g] = True

        best_groups = selected_groups[:k+1]

        best_set = [np.nonzero(groups == g)[0] for g in best_groups]
        best_set = np.hstack(best_set)

        best_w = problem.opt(best_set)
        last_w = best_w

        best_score = problem.score(best_set, best_w)
        c = np.sum(costs[best_groups])

        #print best_w
        #print best_score

        timestamp = time.time() - t0
        sequence.append((best_score, c, g, best_set, best_groups, best_w, timestamp))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                                       ('selected', object), ('selected_groups', object),
                                       ('w', object), ('time', np.float)])
