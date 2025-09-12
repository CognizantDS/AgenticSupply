"""
References :
    https://www.pywhy.org/dowhy/v0.13/example_notebooks/gcm_rca_microservice_architecture.html#Appendix:-Data-generation-process

Examples :
    python -m agentic_supply.data.data_generation
"""

import pandas as pd
import numpy as np
from scipy.stats import truncexpon, halfnorm


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


if __name__ == "__main__":
    normal_data = create_observed_latency_data(unobserved_intrinsic_latencies_normal(10000))
    outlier_data = create_observed_latency_data(unobserved_intrinsic_latencies_anomalous(1000))
    normal_data.to_csv("./src/agentic_supply/data/microservices_latencies_data.csv", index=False)
    outlier_data.to_csv("./src/agentic_supply/data/microservices_latencies_outlier_data.csv", index=False)
    print(f"generated data with shape : normal_data {normal_data.shape}, outlier_data {outlier_data.shape}")
