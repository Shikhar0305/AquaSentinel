
from src.data_loader import load_dataset
from src.inspect_data import inspect_dataset
from src.preprocessing import create_features, scenario_split
from train import run_training


def main():
    print("=" * 60)
    print("AQUASENTINEL - PIPELINE LEAK DETECTION PIPELINE")
    print("=" * 60)

    # Step 1: Load the simulated zone data (simulates it the first time)
    dataset = load_dataset()

    # Step 2: Inspect the dataset
    inspect_dataset(dataset)

    # Step 3: Create leak detection features
    featured_data = create_features(dataset)

    print("\nFeatured dataset shape:", featured_data.shape)

    # Step 4: Split by scenario
    X_train, X_test, y_train, y_test = scenario_split(
        featured_data,
        test_ratio=0.3
    )

    # Step 5: Train, compare, and register models
    results, registered_model = run_training(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nPipeline completed successfully!")
    print("Registered model:", registered_model.name)
    print("Model version:", registered_model.version)


if __name__ == "__main__":
    main()
