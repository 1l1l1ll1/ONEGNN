import unittest
import contextlib
import os

import oneflow as flow
import oneflow.unittest


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    From: https://stackoverflow.com/questions/2059482/temporarily-modify-the-current-processs-environment
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]

def run_testcase_with_sep_compile(test_case_cls):
    new_cls = type("SeparationCompile_" + test_case_cls.__name__, (test_case_cls,), {})
    with modified_environ(ONEFLOW_LAZY_COMPILE_MODE="rank_per_process", ENABLE_LOGICAL_CHAIN="1"):
        unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(new_cls))

@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n4d()
class TestSeparationCompile(oneflow.unittest.TestCase):
    def test_test_alexnet_auto_parallel(test_case):
        from oneflow.test.graph.test_alexnet_auto_parallel import TestAlexnetAutoParallel
        run_testcase_with_sep_compile(TestAlexnetAutoParallel)
    
    def test_comb1to2d(test_case):
        from oneflow.test.graph.test_comb1to2d import TestLazyAllSbpCombinationTesting
        run_testcase_with_sep_compile(TestLazyAllSbpCombinationTesting)
    
    def test_graph_pipeline(test_case):
        from oneflow.test.graph.test_graph_pipeline import TestGraphPipeline
        run_testcase_with_sep_compile(TestGraphPipeline)
    
    def test_graph_zero(test_case):
        from oneflow.test.graph.test_graph_zero import TestLinearTrainGraph2DWithZeRO
        run_testcase_with_sep_compile(TestLinearTrainGraph2DWithZeRO)
    
    def test_graph_clip_grad_norm(test_case):
        from oneflow.test.graph.test_graph_clip_grad_norm import TestGraphClipGradNorm
        run_testcase_with_sep_compile(TestGraphClipGradNorm)

    # TODO(): add grad acc and activation checkpoint test cases

if __name__ == "__main__":
    unittest.main()
