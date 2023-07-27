
import math

class AutoScaleManagerEvaluator():
    def __init__(self) -> None:
        pass

    def evaluation(self, plan, threshold, observation):
        assert len(plan) == len(observation)

        under_provisioning_duration = 0
        under_provisioning_value = 0
        under_provisioning_intensity = 0 
        under_utilization_duration = 0
        count = 0
        for num_instances, observed_resource_usage in zip(plan, observation):
            capacity = num_instances * threshold
            if capacity < observed_resource_usage:
                under_provisioning_duration +=1 
                under_provisioning_value += ((observed_resource_usage/num_instances - threshold) ** 2)
            if math.ceil(observed_resource_usage/threshold) < num_instances:
                under_utilization_duration += 1
            count+=1
        if under_provisioning_duration != 0:
            under_provisioning_intensity = math.sqrt(under_provisioning_value / under_provisioning_duration)
        else:
            under_provisioning_intensity =0

        
        return under_provisioning_duration, under_provisioning_intensity, under_utilization_duration