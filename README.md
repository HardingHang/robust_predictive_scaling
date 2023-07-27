# Robust Auto-Scaling with Probabilistic Workload Forecasting for Cloud Databases

## Project Description
This repository contains the core code implementation of the research paper titled "Robust Auto-Scaling with Probabilistic Workload Forecasting for Cloud Databases." The project aims to address the challenging problem of auto-scaling cloud databases while considering workload uncertainties and providing robust performance.

The code in this repository represents the initial implementation of the auto-scaling mechanism proposed in the research paper. While it forms the core foundation, it is essential to acknowledge that the code is a work in progress and requires further improvements and optimizations.

## Folder Structure

~~~
.
├── forecaster         # probabilistic workload forecasting models
├── scaler_manager     # auto-scaling strategies
├── tests              # Experiments
└── README.md 
~~~


## Requirements
- ```GluonTS ```
- ```scipy ```


## Usage
To run, you need workload trace data prepared (```data/```).
For easy usage, you can refer to the code in  ```tests```.
