import pandas as pd

# Load dataset
df = pd.read_csv("teen_phone_addiction_dataset.csv")

# Rename columns for simplicity
df = df.rename(columns={
    'Screen_Time_Before_Bed': 'ScreenTime',
    'Time_on_Social_Media': 'SocialMedia'
})

# 🔍 Inspect raw Addiction_Level values
print("🔍 Raw Addiction_Level value counts:")
print(df["Addiction_Level"].value_counts(dropna=False))
print("\n🔍 Unique values in Addiction_Level:")
print(df["Addiction_Level"].unique())