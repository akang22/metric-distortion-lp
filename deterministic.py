import itertools
import math
import numbers

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, getSolver, PULP_CBC_CMD
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager, Lock, Queue, Process

from cache import Cache

def sample_deterministic_distortion(m, n):
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
                        sol, dist = solve_deterministic(m, n, preference_lists, opt, w)
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

def solve_deterministic_distortion(m, n, use_cache=True, cache_name="deterministic"):
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

def solve_deterministic(m, n, preference_lists, opt, w, debug=False):
    """
    Solve the LP problem given the number of candidates, voters, and their preferences.
    Runtime: O(nm) variables, O(n^2m^2) clauses

    Args:
        m (int): Number of candidates.
        n (int): Number of voters.
        preference_lists (list of lists): A list of n preference lists, each of length m.
        opt (int): Forced optimal candidate.
        w (int): Chosen winner with preference lists.

    Returns:
        solution: A dictionary with the solution values for the variables.
        distortion: Objective value achieved.
    """
    # Create the LP problem
    model = LpProblem(name="distortion", sense=LpMaximize)

    # Candidates and voters indices
    candidates = list(range(m))  # Candidate indices: 0, 1, ..., m-1
    voters = list(range(n))      # Voter indices: 0, 1, ..., n-1

    # Decision variables d[v][x] for each voter v and candidate x
    d = LpVariable.dicts("d", (voters, candidates), lowBound=0)

    # Objective function: Maximize sum of d[v][w] for all v, w
    model += lpSum(d[v][w] for v in voters)

    # Constraints
    # Triangle Inequality: d[v][x] <= d[v'][x] + d[v'][y] + d[v][y]
    for x in candidates:
        for y in candidates:
            for v1 in voters:
               for v2 in voters:
                    if x != y and v1 != v2:
                        model += (
                            d[v1][x] <= d[v2][x] + d[v2][y] + d[v1][y],
                            f"triangle {v1} {v2} {x} {y}"
                        )

    # Consistency: d[v][x] <= d[v][y] if x >v y (x comes before y in v's preference list)
    for v in voters:
        for i, x in enumerate(preference_lists[v]):
            for y in preference_lists[v][i+1:]:
                model += (
                    d[v][x] <= d[v][y],
                    f"consistency {v} {x} {y}"
                )

    # Normalization: Sum_{v} d[v][x] == 1 for all x
    model += (
        lpSum(d[v][opt] for v in voters) == 1,
        f"normalization"
    )

    # Optimality of x*: Sum_{v} d[v][x] >= 1 for all x
    for x in candidates:
        if x != opt:
            model += (
                lpSum(d[v][x] for v in voters) >= 1,
                f"optimality {x}"
            )

    # Solve the model
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

