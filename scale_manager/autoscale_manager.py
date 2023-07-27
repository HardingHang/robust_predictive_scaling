import math
from scipy.optimize import linprog
import pulp
import numpy as np
import time

from gluonts.model.forecast import QuantileForecast

pulp.LpSolverDefault.msg = 0

class AutoScaleManager():
    def __init__(self):
        self.prob = pulp.LpProblem("Auto-Scale Optimization Problem", pulp.LpMinimize)
    
    def _moving_window_aggregate(self, array, offset, window_size, aggregate_type):
        result = []
        for i in range(0, offset):
            window = array[-(offset+i)-window_size : -(offset+i)]
        
            if aggregate_type == 'max':
                aggregate = np.max(window)
            elif aggregate_type == 'weighted average':
                weights = np.power(0.5, np.arange(window_size-1, -1, -1) / 6)  # 指数衰减权重，半衰期为6个step
                aggregate = np.average(window, weights=weights)
            
            result.append(aggregate)
    
        return result


    def reactive_solution(self, observation, thresholds, metric = "weighted average"):
        num_time_intervals = len(thresholds)
        start_time = time.perf_counter()
        result = self._moving_window_aggregate(observation, num_time_intervals, 6, metric)
        prob = self.prob.deepcopy()
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(num_time_intervals)]
        prob += pulp.lpSum(x)
        for j in range(num_time_intervals):
            prob += thresholds[j] * x[j] >= result[j]


        status = prob.solve()
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        plan = None
        if status == pulp.LpStatusOptimal:
            plan = [int(pulp.value(i)) for i in x]
            # print("Optimal solution found.")
            # print(f"Objective function value = {int(pulp.value(prob.objective))}")
            # print(f"Plan = {plan}")
        else:
            print("Unable to find optimal solution.")

        return plan
    def hybrid_forecaster_solution(self, forecasts, thresholds):
        num_time_intervals = len(thresholds)

        point_forecasts = forecasts

        prob = self.prob.deepcopy()
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(num_time_intervals)]
        prob += pulp.lpSum(x)
        for j in range(num_time_intervals):
            prob += thresholds[j] * x[j] >= point_forecasts[j]


        status = prob.solve()

        plan = None
        if status == pulp.LpStatusOptimal:
            plan = [int(pulp.value(i)) for i in x]
            # print("Optimal solution found.")
            # print(f"Objective function value = {int(pulp.value(prob.objective))}")
            # print(f"Plan = {plan}")
        else:
            print("Unable to find optimal solution.")

        return plan

    def basic_solution(self, forecasts: QuantileForecast, thresholds):
        num_time_intervals = len(thresholds)

        point_forecasts = forecasts.mean

        prob = self.prob.deepcopy()
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(num_time_intervals)]
        prob += pulp.lpSum(x)
        for j in range(num_time_intervals):
            prob += thresholds[j] * x[j] >= point_forecasts[j]


        status = prob.solve()

        plan = None
        if status == pulp.LpStatusOptimal:
            plan = [int(pulp.value(i)) for i in x]
            # print("Optimal solution found.")
            # print(f"Objective function value = {int(pulp.value(prob.objective))}")
            # print(f"Plan = {plan}")
        else:
            print("Unable to find optimal solution.")

        return plan
    
    def robust_solution(self, forecasts: QuantileForecast, quantile, thresholds):
    
        num_time_intervals = len(thresholds)
        start_time = time.perf_counter()
        point_forecasts = forecasts.quantile(quantile)

        prob = self.prob.deepcopy()
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(num_time_intervals)]
        prob += pulp.lpSum(x)
        for j in range(num_time_intervals):
            prob += thresholds[j] * x[j] >= point_forecasts[j]


        status = prob.solve()
        end_time = time.perf_counter()
        execution_time = end_time - start_time


        plan = None
        if status == pulp.LpStatusOptimal:
            plan = [int(pulp.value(i)) for i in x]
            # print("Optimal solution found.")
            # print(f"Objective function value = {int(pulp.value(prob.objective))}")
            # print(f"Plan = {plan}")
        else:
            print("Unable to find optimal solution.")

        return plan
    
    def _uncertainty_measurement(self, quantiles_level, forecasts):
        length = len(forecasts[0])
        uncertainty_level_list = []
        for i in range(length):
            uncertainty_measurement = 0
            predicted_quantiles = [row[i] for row in forecasts]
            mean_forecast_idx = quantiles_level.index(0.5)
            mean_forecast = predicted_quantiles[mean_forecast_idx]
            for i in range(len(quantiles_level)):
                quantile_level = quantiles_level[i]
                if predicted_quantiles[i] < mean_forecast:
                    uncertainty_measurement = uncertainty_measurement+(mean_forecast-predicted_quantiles[i])*quantile_level
                else:
                    uncertainty_measurement = uncertainty_measurement+(predicted_quantiles[i]-mean_forecast)*(1- quantile_level)
            uncertainty_level_list.append(uncertainty_measurement)
        return uncertainty_level_list
    
    def adaptive_robust_solution(self, forecasts: list[QuantileForecast], quantiles, thresholds, uncertainty_threshold=4):
        num_time_intervals = len(thresholds)
        start_time = time.perf_counter()
        # uncertainty_threshold = 10 # alibaba deeepar 6
        uncertainty_threshold = uncertainty_threshold # alibaba tft 4
        # uncertainty_threshold = 10 # google tft 10
        # uncertainty_threshold = 5 # google deeepar 5
    
        quantiles_level = [float(key) for key in forecasts.forecast_keys]
        quantile_forecasts = [forecasts.quantile(i) for i in quantiles_level]
        uncertainty_level_list = self._uncertainty_measurement(quantiles_level, quantile_forecasts)
        #print(uncertainty_level_list)
        prob = self.prob.deepcopy()
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(num_time_intervals)]
        prob += pulp.lpSum(x)
        for j in range(num_time_intervals):
            predicted_quantiles = [row[j] for row in quantile_forecasts]
            if uncertainty_level_list[j] <= uncertainty_threshold:
                prob += thresholds[j] * x[j] >= predicted_quantiles[quantiles_level.index(quantiles[0])]
            else:
                prob += thresholds[j] * x[j] >= predicted_quantiles[quantiles_level.index(quantiles[1])]

        status = prob.solve()
        end_time = time.perf_counter()
        execution_time = end_time - start_time


        plan = None
        if status == pulp.LpStatusOptimal:
            plan = [int(pulp.value(i)) for i in x]
            # print("Optimal solution found.")
            # print(f"Objective function value = {int(pulp.value(prob.objective))}")
            # print(f"Plan = {plan}")
        else:
            print("Unable to find optimal solution.")

        return plan
