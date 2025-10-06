import argparse
import os
import random
from datetime import datetime, timedelta
import pandas as pd


def generate_user_ids(num_users: int) -> list[str]:
	return [f"USER_{i:05d}" for i in range(1, num_users + 1)]


def generate_dates(num_days: int, start_date: str | None) -> list[str]:
	if start_date:
		start = datetime.strptime(start_date, "%Y-%m-%d")
	else:
		start = datetime(2023, 1, 1)
	return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]


def generate_modality_df(users: list[str], dates: list[str], col_name: str, low: int, high: int, seed: int) -> pd.DataFrame:
	random.seed(seed)
	data_rows: list[dict[str, object]] = []
	for user in users:
		for date_str in dates:
			value = random.randint(low, high)
			data_rows.append({"user": user, "date": date_str, col_name: value})
	return pd.DataFrame(data_rows)


def write_csv(df: pd.DataFrame, out_dir: str, filename: str) -> str:
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, filename)
	df.to_csv(path, index=False)
	return path


def main():
	parser = argparse.ArgumentParser(description="Generate synthetic sample datasets for testing.")
	parser.add_argument("--users", type=int, default=50, help="Number of unique users to generate")
	parser.add_argument("--days", type=int, default=30, help="Number of sequential days to generate")
	parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date YYYY-MM-DD")
	parser.add_argument("--out", type=str, default="sample_data", help="Output directory")
	parser.add_argument("--prefix", type=str, default="large", help="Filename prefix to avoid overwrites")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	args = parser.parse_args()

	users = generate_user_ids(args.users)
	dates = generate_dates(args.days, args.start_date)

	# Define ranges similar to existing samples and tests
	logon_df = generate_modality_df(users, dates, "logon", low=0, high=30, seed=args.seed + 1)
	file_df = generate_modality_df(users, dates, "file", low=0, high=20, seed=args.seed + 2)
	email_df = generate_modality_df(users, dates, "email", low=0, high=200, seed=args.seed + 3)
	device_df = generate_modality_df(users, dates, "device", low=0, high=15, seed=args.seed + 4)
	http_df = generate_modality_df(users, dates, "http", low=0, high=10000, seed=args.seed + 5)

	out_dir = args.out
	prefix = f"{args.prefix}_{args.users}u_{args.days}d"

	paths = {}
	paths["logon"] = write_csv(logon_df, out_dir, f"logon_{prefix}.csv")
	paths["file"] = write_csv(file_df, out_dir, f"file_{prefix}.csv")
	paths["email"] = write_csv(email_df, out_dir, f"email_{prefix}.csv")
	paths["device"] = write_csv(device_df, out_dir, f"device_{prefix}.csv")
	paths["http"] = write_csv(http_df, out_dir, f"http_{prefix}.csv")

	# Also produce combined per-row dataset with all modalities for convenience
	combined = logon_df.merge(file_df, on=["user", "date"])\
		.merge(email_df, on=["user", "date"])\
		.merge(device_df, on=["user", "date"])\
		.merge(http_df, on=["user", "date"])\
		.sort_values(["user", "date"])\
		.reset_index(drop=True)
	paths["combined"] = write_csv(combined, out_dir, f"sample_{prefix}.csv")

	print("Generated files:")
	for k, v in paths.items():
		print(f"- {k}: {v}")


if __name__ == "__main__":
	main()

