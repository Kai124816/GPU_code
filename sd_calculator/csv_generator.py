import csv
import random

# Configuration
output_file = "random_numbers.csv"
num_values = 50000

# Generate 50,000 random float numbers between 0 and 1
random_numbers = [random.uniform(0, 1) for _ in range(num_values)]

# Write to CSV file (one column)
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["value"])  # header
    for value in random_numbers:
        writer.writerow([value])

print(f"Generated {num_values} random numbers and saved to '{output_file}'.")
