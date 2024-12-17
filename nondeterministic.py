import itertools
import math
import numbers

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, getSolver, PULP_CBC_CMD
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager, Lock, Queue, Process

from cache import Cache

def sample_non_deterministic_distortion(m, n):
    """
    Args:
        m (int): Number of candidates.
        n (int): Number of voters.
    """

# we require also the winner not be dominated
def plists_gen(m, n):
    generator = itertools.combinations_with_replacement(itertools.permutations(list(range(m)), m), n)
    for plists in generator:
        dominated_candidates = set()
        for candidate in range(m):
            dset = set(range(m))
            for l in plists:
                dset = dset.intersection(l[0:l.index(candidate)])
                if len(dset) == 0:
                    break
            if len(dset) != 0:
                dominated_candidates.add(candidate)
        yield plists, dominated_candidates


def solve_worker_chunk(chunk_args):
    """
    Worker function to solve a chunk of tasks for deterministic distortion.
    """
    global q
    m, n, chunk, use_cache, cache_file = chunk_args
    cache = Cache(cache_file)
    results = []

    for i, (preference_lists, dominated_list) in chunk:
        key = f"{m} {n} {i}"
        q.put(SENTINEL)

        def solve():
            min_dist = math.inf
            min_vars = {}
            errs = []
            chosen_w = math.inf
            for w in range(m):
                max_dist_w = 1
                max_vars_w = {}
                for opt in range(m):
                    if opt == w or opt in dominated_list or w in dominated_list:
                        continue
                    try:
                        sol, dist = solve_non_deterministic(m, n, preference_lists)
                        if dist > max_dist_w:
                            max_dist_w = dist
                            max_vars_w = sol
                    except Exception as e:
                        errs.append(f"({m},{n},{preference_lists},0,{w})")
                        continue
                if min_dist > max_dist_w:
                    min_dist = max_dist_w
                    min_vars = max_vars_w
                    chosen_w = w
            return (min_dist, min_vars, chosen_w, errs)

        results.append((cache.check_cache(key, solve, use_cache=use_cache, persist_cache=(i % 100 == 0) or i == len(chunk))))

    return results

SENTINEL = 1

def listener(num_iters):
    global q
    pbar = tqdm(total = num_iters )
    for item in iter(q.get, None):
        pbar.update()

def solve_non_deterministic_distortion(m, n, use_cache=True, cache_name="deterministic"):
    """For a given n, m, we want to solve all LPs like this.
    
    Thus, this is a total of O(LP(m^2n^2, nm) * m^2 * (m!)^n). 
    
    To keep things small, we restrict 1 < m < 6 (m = 1 is not interesting).
    we also assume opt = 0 by symmetry and w != opt otherwise distortion 1 is achieved.
    
    Args:
        m (int): Number of candidates.
        n (int): Number of voters.

    Returns:
        distortions: A list of all distortions achieved.
        max_distortion: Maximum distortion achieved for a preference list.
        max_vars: Distance values in the maximum distortion achieved.
        errs: Errored inputs
    """
    # note: not space efficient. Will always "use up" permutation space.
    # also bad with multithreading
    cache_file = f"caches/{cache_name}.json"
    preflist_gen = list(plists_gen(m, n))
    global q
    q = Queue()
    mfact, nfact = math.factorial(m), math.factorial(n)
    num_iters = math.prod([mfact + i for i in range(n)]) // nfact
    proc = Process(target=listener, args=(num_iters,))
    proc.start()

    i = 0
    dists = []
    assoc_info = []
    ret_dist = 1
    ret_vars = {}
    ret_errs = []

    # Divide tasks into chunks
    num_workers = multiprocessing.cpu_count()
    chunk_size = math.ceil(len(preflist_gen) / num_workers)
    chunks = [(m, n, list(enumerate(preflist_gen[i:i + chunk_size], start=i)), use_cache, cache_file)
              for i in range(0, len(preflist_gen), chunk_size)]

    # Use multiprocessing to process chunks
    with Pool(processes=num_workers) as pool:
        chunk_results = list(pool.imap(solve_worker_chunk, chunks))
    q.put(None)
    proc.join()

    for chunk in chunk_results:
        for dist, vars, chosen_w, errs in chunk:
            dists.append(dist)
            ret_errs += errs
            if dist > ret_dist:
                ret_vars = vars
                ret_dist = dist
    return dists, assoc_info, ret_dist, ret_vars, ret_errs

def solve_non_deterministic(m, n, preference_lists, debug=False):
    """
    Solve the LP problem given the number of candidates, voters, and their preferences.
    Runtime: O(nm) variables, O(n^2m^2) clauses

    Args:
        m (int): Number of candidates.
        n (int): Number of voters.
        preference_lists (list of lists): A list of n preference lists, each of length m.
        opt (int): Forced optimal candidate.

    Returns:
        solution: A dictionary with the solution values for the variables.
        distortion: Objective value achieved.
    """
    # Create the LP problem
    model = LpProblem(name="distortion", sense=LpMaximize)

    # Candidates and voters indices
    candidates = list(range(m))  # Candidate indices: 0, 1, ..., m-1
    voters = list(range(n))      # Voter indices: 0, 1, ..., n-1

      # Decision variables
    Z = LpVariable("Z", lowBound=0)  # Objective variable
    d = LpVariable.dicts("d", (voters, candidates), lowBound=0)  # Distances
    o = LpVariable.dicts("o", candidates, lowBound=0)  # Optimality distribution

    # Objective function
    model += Z

    # Constraint 1: Z ≤ Σ d[v, X] for all X
    for X in candidates:
        model += (Z <= lpSum(d[v][X] for v in voters), f"Z_bound_{X}")

    # Constraint 2: d[v, X] ≤ d[v', X] + d[v', Y] + L[Y, v] (Triangle Inequality)
    for v in voters:
        for v_prime in voters:
            for X in candidates:
                for Y in candidates:
                    model += (
                        d[v][X] <= d[v_prime][X] + d[v_prime][Y] + d[v][Y],
                        f"triangle_ineq_{v}_{v_prime}_{X}_{Y}"
                    )

    # Constraint 3: L[X, v] ≤ L[Y, v] for X ≥v Y (Consistency)
    for v in voters:
        for X in candidates:
            for Y in candidates:
                if X >= Y:  # Assuming we have a way to check if X >=v Y
                    model += (
                        d[v][X] <= d[v][Y],
                        f"consistency_{v}_{X}_{Y}"
                    )

    # Constraint 4: Σ_v Σ_X o_X * L[X, v] = 1 (Normalization)
    # todo: we do not support products, need to use a different library
    model += (
        lpSum(o[X] * d[v][X] for v in voters for X in candidates) == 1,
        "normalization"
    )

    # Constraint 5: Σ_v L[x, v] ≥ 1 for all x (Optimality of x*)
    for x in candidates:
        model += (
            lpSum(d[v][x] for v in voters) >= 1,
            f"optimality_{x}"
        )

    # Constraint 6: Σ_x o_x = 1 (Definition of optimality distribution)
    model += (
        lpSum(o[x] for x in candidates) == 1,
        "opt_dist"
    )

    status = model.solve(PULP_CBC_CMD(msg=debug))

    # Check the status of the solution
    if status != 1:
        raise ValueError("The problem could not be solved.")

    # Extract and return the solution
    solution = {
        f"{(v, x)}": d[v][x].varValue for v in voters for x in candidates
    }
    distortion = model.objective.value()
    if not isinstance(distortion, numbers.Number):
        raise ValueError("The problem could not be solved.")
    return solution, distortion

