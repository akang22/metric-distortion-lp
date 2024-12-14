import itertools
import numbers

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, getSolver, PULP_CBC_CMD

from cache import Cache

def sample_deterministic_distortion(m, n):
    """
    Args:
        m (int): Number of candidates.
        n (int): Number of voters.
    """

def plists_gen(m, n, undominated_candidate = 0):
    generator = itertools.product(itertools.permutations(list(range(m)), m), repeat=n)
    while plists := next(generator, False):
        # check to see if candidate is ever "dominated" by another, making it impossible for it to be optimal
        dset = set(range(m))
        for l in plists:
            dset = dset.intersection(l[0:l.index(undominated_candidate)])
            if len(dset) == 0:
                break
        if len(dset) == 0:
            yield plists
        else:
            continue
    yield False


def solve_deterministic_distortion(m, n, use_cache=True):
    """For a given n, m, we want to solve all LPs like this.
    
    Thus, this is a total of O(LP(m^2n^2, nm) * m^2 * (m!)^n). 
    
    To keep things small, we restrict 1 < m < 6 (m = 1 is not interesting).
    we also assume opt = 0 by symmetry and w != opt otherwise distortion 1 is achieved.
    
    Args:
        m (int): Number of candidates.
        n (int): Number of voters.

    Returns:
        distortions: A list of all distortions achieved.
        max_distortion: Maximum distortion achieved.
        max_vars: Distance values in the maximum distortion achieved.
        errs: Errored inputs
    """
    # note: not space efficient. Will always "use up" permutation space.
    # also bad with multithreading
    cache = Cache("caches/deterministic.json");
    preflist_gen = plists_gen(m, n)
    i = 0
    dists = []
    ret_dist = 1
    ret_vars = {}
    ret_errs = []
    while preference_lists := next(preflist_gen):
        # TODO: put a thread here.
        key = f"{m} {n} {i}"
        def solve():
            ret = []
            max_dist = 1
            max_vars = {}
            errs = []
            for w in range(1, m):
                try:
                    sol, dist = solve_deterministic(m, n, preference_lists, 0, w)
                    ret.append(dist)
                    if max_dist < dist:
                        max_dist = dist
                        max_vars = sol
                except Exception as e:
                    errs.append(f"({m},{n},{preference_lists},0,{w})")
                    continue
            return (ret, max_dist, max_vars, errs)

        cur_dists, max_dist, max_vars, errs = cache.check_cache(key, solve)
        dists += cur_dists
        ret_errs += errs
        if max_dist > ret_dist:
            ret_vars = max_vars
            ret_dist = max_dist
        i += 1
    return dists, ret_dist, ret_vars, ret_errs

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
    status = model.solve(PULP_CBC_CMD(msg=0))

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

