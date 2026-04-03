import csv
import sys
import matplotlib.pyplot as plt

filename = "build/erosion_profiles.csv"

# Allow optional filename argument
if len(sys.argv) > 1:
    filename = sys.argv[1]

phi = []
columns = []
headers = []

with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Read header
    headers = next(reader)
    
    # Prepare column containers
    for _ in headers:
        columns.append([])

    # Read rows
    for row in reader:
        for i, value in enumerate(row):
            columns[i].append(float(value))

# First column = phi
phi = columns[0]

# Plot remaining columns
for i in range(1, len(columns)):
    plt.plot(phi, columns[i], label=headers[i])

plt.xlabel("phi")
plt.ylabel("profile value")
plt.title("Erosion Profiles")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
