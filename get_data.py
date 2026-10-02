from datasets import load_dataset

ds = load_dataset("papluca/language-identification")
ds["train"].to_pandas().to_csv("data/train.csv", index=False)
ds["test"].to_pandas().to_csv("data/test.csv", index=False)
print("Saved!")