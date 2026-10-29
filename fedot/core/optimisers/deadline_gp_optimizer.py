"""Experimental opt-in population deadline using GOLEM's existing joblib pool."""
from functools import partial
from inspect import signature
import math
import time
from multiprocessing import TimeoutError as PoolTimeout

from joblib import Parallel, delayed
from golem.core.log import Log
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError


class PopulationDeadlineExceeded(EvaluationAttemptsError):
    """The shared search deadline expired while evaluating a population."""


class DeadlinePopulationDispatcher(MultiprocessingDispatcher):
    def evaluate_population(self, individuals):
        pending, skipped = self.split_individuals_to_evaluate(individuals)
        if not pending:
            return skipped
        if self.timer is None or self.timer.timeout is None:
            raise ValueError("population deadline requires a finite search timeout")
        remaining = self.timer.timeout.total_seconds() - self.timer.seconds_from_start
        if not math.isfinite(remaining) or remaining <= 0:
            raise PopulationDeadlineExceeded("search deadline expired")
        if self._n_jobs < 2:
            raise ValueError("population deadline requires at least two joblib workers")
        if 'return_as' not in signature(Parallel).parameters:
            raise RuntimeError("population deadline requires joblib >= 1.3")
        evaluate = partial(self.evaluate_single, logs_initializer=Log().get_parameters())
        deadline = time.monotonic() + remaining
        results = []
        expired = False
        try:
            with Parallel(n_jobs=self._n_jobs, timeout=remaining,
                          return_as="generator", backend="loky") as parallel:
                if parallel._effective_n_jobs() < 2:
                    raise RuntimeError("population deadline requires a process pool")
                stream = parallel(delayed(evaluate)(item.graph, item.uid) for item in pending)
                try:
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise PoolTimeout()
                        # Joblib's timeout applies to the next awaited batch.
                        parallel.timeout = remaining
                        try:
                            result = next(stream)
                        except StopIteration:
                            break
                        if time.monotonic() >= deadline:
                            raise PoolTimeout()
                        results.append(result)
                finally:
                    stream.close()
        except PoolTimeout:
            expired = True
        completed = self.apply_evaluation_results(pending[:len(results)], results) + skipped
        self.population_evaluation_info(evaluated_pop_size=len(completed), pop_size=len(individuals))
        if not completed:
            if expired:
                raise PopulationDeadlineExceeded("parallel population exceeded search deadline")
            # Do not enter the original unbounded sequential retry fallback.
            raise EvaluationAttemptsError("population produced no valid fitness within search deadline")
        return completed


class DeadlineEvoGraphOptimizer(EvoGraphOptimizer):
    def _initial_population(self, evaluator):
        try:
            super()._initial_population(evaluator)
        except PopulationDeadlineExceeded:
            if not self.best_individuals:
                raise

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original = self.eval_dispatcher
        if isinstance(original, MultiprocessingDispatcher):
            self.eval_dispatcher = DeadlinePopulationDispatcher(
                adapter=original._adapter,
                n_jobs=original._n_jobs,
                graph_cleanup_fn=original._cleanup,
                delegate_evaluator=original._delegate_evaluator,
            )
