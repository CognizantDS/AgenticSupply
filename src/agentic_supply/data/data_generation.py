"""
References :
    https://www.pywhy.org/dowhy/v0.13/example_notebooks/gcm_rca_microservice_architecture.html#Appendix:-Data-generation-process
    https://www.pywhy.org/dowhy/v0.13/example_notebooks/gcm_counterfactual_medical_dry_eyes.html#Appendix:-What-the-tele-app-uses-internally.-Data-generation-of-the-patients'-log

Examples :
    python -m agentic_supply.data.data_generation --data_name microservices_latencies
    python -m agentic_supply.data.data_generation --data_name medical_case
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import truncexpon, halfnorm, bernoulli, norm, uniform
from random import randint
from importlib.resources import files, as_file

from agentic_supply import data

P_1 = 0.2
P_2 = 0.15


def create_observed_latency_data(unobserved_intrinsic_latencies) -> pd.DataFrame:
    observed_latencies = {}
    observed_latencies["Product DB"] = unobserved_intrinsic_latencies["Product DB"]
    observed_latencies["Customer DB"] = unobserved_intrinsic_latencies["Customer DB"]
    observed_latencies["Order DB"] = unobserved_intrinsic_latencies["Order DB"]
    observed_latencies["Shipping Cost Service"] = unobserved_intrinsic_latencies["Shipping Cost Service"]
    observed_latencies["Caching Service"] = (
        np.random.choice([0, 1], size=(len(observed_latencies["Product DB"]),), p=[0.5, 0.5]) * observed_latencies["Product DB"]
        + unobserved_intrinsic_latencies["Caching Service"]
    )
    observed_latencies["Product Service"] = (
        np.maximum(
            np.maximum(observed_latencies["Shipping Cost Service"], observed_latencies["Caching Service"]),
            observed_latencies["Customer DB"],
        )
        + unobserved_intrinsic_latencies["Product Service"]
    )
    observed_latencies["Auth Service"] = observed_latencies["Customer DB"] + unobserved_intrinsic_latencies["Auth Service"]
    observed_latencies["Order Service"] = observed_latencies["Order DB"] + unobserved_intrinsic_latencies["Order Service"]
    observed_latencies["API"] = (
        observed_latencies["Product Service"]
        + observed_latencies["Customer DB"]
        + observed_latencies["Auth Service"]
        + observed_latencies["Order Service"]
        + unobserved_intrinsic_latencies["API"]
    )
    observed_latencies["www"] = observed_latencies["API"] + observed_latencies["Auth Service"] + unobserved_intrinsic_latencies["www"]
    observed_latencies["Website"] = observed_latencies["www"] + unobserved_intrinsic_latencies["Website"]

    return pd.DataFrame(observed_latencies)


def unobserved_intrinsic_latencies_normal(num_samples):
    return {
        "Website": truncexpon.rvs(size=num_samples, b=3, scale=0.2),
        "www": truncexpon.rvs(size=num_samples, b=2, scale=0.2),
        "API": halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        "Auth Service": halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        "Product Service": halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        "Order Service": halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        "Shipping Cost Service": halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        "Caching Service": halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
        "Order DB": truncexpon.rvs(size=num_samples, b=5, scale=0.2),
        "Customer DB": truncexpon.rvs(size=num_samples, b=6, scale=0.2),
        "Product DB": truncexpon.rvs(size=num_samples, b=10, scale=0.2),
    }


def unobserved_intrinsic_latencies_anomalous(num_samples):
    return {
        "Website": truncexpon.rvs(size=num_samples, b=3, scale=0.2),
        "www": truncexpon.rvs(size=num_samples, b=2, scale=0.2),
        "API": halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        "Auth Service": halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        "Product Service": halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        "Order Service": halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        "Shipping Cost Service": halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        "Caching Service": 2 + halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
        "Order DB": truncexpon.rvs(size=num_samples, b=5, scale=0.2),
        "Customer DB": truncexpon.rvs(size=num_samples, b=6, scale=0.2),
        "Product DB": truncexpon.rvs(size=num_samples, b=10, scale=0.2),
    }


def create_unobserved_medical_data():
    n_unobserved = 10000
    unobserved_data = {
        "N_T": np.array([randint(0, 2) for p in range(n_unobserved)]),
        "N_vision": np.random.uniform(0.4, 0.6, size=(n_unobserved,)),
        "N_C": bernoulli.rvs(0.01, size=n_unobserved),
    }
    return unobserved_data


def create_observed_medical_data(unobserved_data):
    observed_medical_data = {}
    observed_medical_data["Condition"] = unobserved_data["N_C"]
    observed_medical_data["Treatment"] = unobserved_data["N_T"]
    observed_medical_data["Vision"] = (
        unobserved_data["N_vision"]
        + (-P_1)
        * (1 - observed_medical_data["Condition"])
        * (1 - observed_medical_data["Treatment"])
        * (2 - observed_medical_data["Treatment"])
        + (2 * P_2)
        * (1 - observed_medical_data["Condition"])
        * (observed_medical_data["Treatment"])
        * (2 - observed_medical_data["Treatment"])
        + (P_2)
        * (1 - observed_medical_data["Condition"])
        * (observed_medical_data["Treatment"])
        * (1 - observed_medical_data["Treatment"])
        * (3 - observed_medical_data["Treatment"])
        + 0 * (observed_medical_data["Condition"]) * (1 - observed_medical_data["Treatment"]) * (2 - observed_medical_data["Treatment"])
        + (-2 * P_2) * (unobserved_data["N_C"]) * (observed_medical_data["Treatment"]) * (2 - observed_medical_data["Treatment"])
        + (-P_2)
        * (observed_medical_data["Condition"])
        * (observed_medical_data["Treatment"])
        * (1 - observed_medical_data["Treatment"])
        * (3 - observed_medical_data["Treatment"])
    )
    return pd.DataFrame(observed_medical_data)


def generate_specific_patient_data(num_samples=1):
    original_vision = np.random.uniform(0.4, 0.6, size=num_samples)
    return create_observed_medical_data(
        {
            "N_T": np.full((num_samples,), 2),
            "N_C": bernoulli.rvs(1, size=num_samples),
            "N_vision": original_vision,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True)
    args = parser.parse_args()
    source = files(data)
    if args.data_name == "microservices_latencies":
        normal_data = create_observed_latency_data(unobserved_intrinsic_latencies_normal(10000))
        outlier_data = create_observed_latency_data(unobserved_intrinsic_latencies_anomalous(1000))
        normal_data.to_csv(source.joinpath(f"{args.data_name}_data.csv"), index=False)
        outlier_data.to_csv(source.joinpath(f"{args.data_name}_outlier_data.csv"), index=False)
        print(f"generated data with shape : normal_data {normal_data.shape}, outlier_data {outlier_data.shape}")
    if args.data_name == "medical_case":
        medical_data = create_observed_medical_data(create_unobserved_medical_data())
        specific_patient_data = generate_specific_patient_data()
        medical_data.to_csv(source.joinpath(f"{args.data_name}_data.csv"), index=False)
        specific_patient_data.to_csv(source.joinpath(f"{args.data_name}_counterfactual_data.csv"), index=False)
        print(f"generated data with shape : medical_data {medical_data.shape}, specific_patient_data {specific_patient_data.shape}")
