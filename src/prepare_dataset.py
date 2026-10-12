import pandas as pd


# Location of the original dataset
input_file = "data/raw/LD2011_2014.txt"

# Location where we will save the smaller dataset
output_file = "data/processed/energy_client_1.csv"


print("Reading dataset...")

df = pd.read_csv(
    input_file,
    sep=";",
    decimal=",",
    parse_dates=[0]
)


# Rename first column to timestamp
df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)


# Select the first electricity client
client_column = df.columns[1]

small_df = df[
    ["timestamp", client_column]
].copy()


# Rename electricity consumption column
small_df.rename(
    columns={client_column: "load"},
    inplace=True
)


# Remove rows where load is zero
small_df = small_df[small_df["load"] > 0]


# Save as CSV
small_df.to_csv(
    output_file,
    index=False
)


print("Dataset prepared successfully!")
print("Client used:", client_column)
print("Number of rows:", len(small_df))
print("Saved to:", output_file)