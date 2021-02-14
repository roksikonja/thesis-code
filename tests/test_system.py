import platform
import sys
import time
import unittest

import pyomo.environ as pyo
import pyomo.opt as pyo_opt


class TestSystem(unittest.TestCase):
    def test_os(self):
        print("Operating System\n")
        print(sys.platform)  # win32 or linux
        print(platform.system(), platform.release())

        time.sleep(0.1)
        self.assertTrue(True)

    @staticmethod
    def create_pyomo_instance():
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
        model.obj = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])
        model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)

        return model

    def test_solver_gurobi(self):
        if sys.platform == "win32":
            model = self.create_pyomo_instance()
            solver = pyo_opt.SolverFactory("gurobi")

            print(solver)
            model.pprint()
            solver.solve(model, tee=True)
            print(model.display())

            time.sleep(0.1)
            self.assertTrue(True)
        else:
            print("Gurobi not installed.")

            time.sleep(0.1)
            self.assertTrue(True)

    def test_solver_mosek(self):
        if sys.platform == "win32":
            model = self.create_pyomo_instance()
            solver = pyo_opt.SolverFactory("mosek")

            print(solver)
            solver.solve(model, tee=True)
            print(model.display())

            time.sleep(0.1)
            self.assertTrue(True)
        else:
            print("Mosek not installed.")

            time.sleep(0.1)
            self.assertTrue(True)

    def test_solver_glpk(self):
        model = self.create_pyomo_instance()
        solver = pyo_opt.SolverFactory("glpk")

        print(solver)
        solver.solve(model, tee=True)
        print(model.display())

        time.sleep(0.1)
        self.assertTrue(True)
