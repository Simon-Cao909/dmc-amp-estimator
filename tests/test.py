import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from estimator import DMCAmpEstimator, CHAN_LIST


# ============================================================
# 1. Generate synthetic detector data
# ============================================================

def generate_fake_data(n_samples=10000, seed=42):
    rng = np.random.default_rng(seed)

    # Generate total event energy
    Etot = rng.uniform(50, 500, size=n_samples)

    # Generate channel fractions.
    # We make these correlated so the missing channels can
    # be learned from the observed channels.
    latent = rng.dirichlet(
        alpha=np.ones(len(CHAN_LIST)) * 2.0,
        size=n_samples
    )

    # Convert fractions into amplitudes
    amps = latent * Etot[:, None]

    # Add a little detector noise
    noise = rng.normal(
        loc=0,
        scale=0.01 * amps + 0.001,
        size=amps.shape
    )

    amps = np.maximum(amps + noise, 0)

    df = pd.DataFrame(
        amps,
        columns=[c + "OFamps" for c in CHAN_LIST]
    )

    return df


# ============================================================
# 2. Generate training and test data
# ============================================================

train_df = generate_fake_data(
    n_samples=20000,
    seed=1
)

test_df = generate_fake_data(
    n_samples=5000,
    seed=2
)


# ============================================================
# 3. Create estimator
# ============================================================

missing_chans = [
    "PAS1",
    "PBS1"
]

estimator = DMCAmpEstimator(
    missing_chans=missing_chans
)


# ============================================================
# 4. Train
# ============================================================

print("Training model...")

estimator.fit(train_df)

print("Training complete!")


# ============================================================
# 5. Evaluate
# ============================================================

print("\nEvaluating model...")

info = estimator.evaluate(test_df)


for channel in estimator.rec:
    print(
        f"{channel}: "
        f"RMSE = {info['rmse'][channel]:.4f}, "
        f"STD = {info['std'][channel]:.4f}"
    )


# ============================================================
# 6. Generate validation plots
# ============================================================

plots = estimator.get_validation_plots(test_df)

plt.show()

# Train original model
estimator = DMCAmpEstimator(
    missing_chans=["PAS1", "PBS1"]
)

estimator.fit(train_df)

print("Original model:")
info = estimator.evaluate(test_df)

for channel in estimator.rec:
    print(
        channel,
        "RMSE:",
        info["rmse"][channel]
    )


# Create a new dataset representing a different detector/domain
transfer_train_df = generate_fake_data(
    n_samples=5000,
    seed=100
)

# Transfer learn
estimator.transfer_learn(transfer_train_df)

print("\nAfter transfer learning:")
info = estimator.evaluate(test_df)

for channel in estimator.rec:
    print(
        channel,
        "RMSE:",
        info["rmse"][channel]
    )