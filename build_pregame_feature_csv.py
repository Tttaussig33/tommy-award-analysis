from pathlib import Path

from pregame_features import build_pregame_feature_csv


def main() -> None:
    input_path = Path("Tommy_Award_Player_Game_Table.csv")
    output_path = Path("Tommy_Award_Pregame_Features.csv")

    feature_df = build_pregame_feature_csv(input_path=input_path, output_path=output_path)

    print(f"Saved {len(feature_df)} rows to {output_path}")
    print("Created leakage-safe pregame columns:")
    print("- season averages before each game")
    print("- last 3 game averages before each game")
    print("- last 5 game averages before each game")
    print("- last 10 game averages before each game")


if __name__ == "__main__":
    main()
