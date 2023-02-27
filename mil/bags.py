# INPUTS
import numpy as np

# BAG CONFIGURATNION PARAMETERS
POSITIVE_CLASS = 1                      # Positive Class Label
BAG_COUNT = 1000                        # Number of Training Bags
VAL_BAG_COUNT = 300                     # Number of Validation Bags
BAG_SIZE = 3                            # Number of Instances in Bag
PLOT_SIZE = 3                           # Number of Bags to plot
ENSEMBLE_AVG_COUNT = 1                  # Number of Models to create and average together


# CREATE BAG
def make_bags(input_data, input_labels, positive_class, bag_count, instance_count):

    # Set Up:
    bags = []
    bag_labels = []

    # Normalize:
    input_data = np.divide(input_data, 255.0)

    positive_count = 0

    # Add Instances to Bags:
    for _ in range(bag_count):
        # Randomly Choose Instances
        index = np.random.choice(input_data.shape[0], instance_count, replace = False)
        instances_data = input_data[index]
        instances_labels = input_labels[index]

        bag_label = 0                   # Default = 0, Negative

        if positive_class in instances_labels:
            bag_label = 1               # Bag is Positive if any instance is positive
            positive_count += 1

        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))
    
    # Print Counts:
    print(f"Positive Bags: {positive_count}")
    print(f"Negative Bags: {bag_count - positive_count}")

    return (list(np.swapaxes(bags, 0 , 1)), np.array(bag_labels))