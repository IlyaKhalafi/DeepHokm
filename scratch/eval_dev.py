import sys
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')

if __name__ == "__main__":
    from deephokm.training.gauntlet_workers import run_team_gauntlet_shards, OpponentSpec
    n = 400
    wins = run_team_gauntlet_shards(
        "checkpoints/dev_only.zip", OpponentSpec(kind="greedy"),
        n_games=n, n_workers=16, seed=99)
    print(f"dev_only: {wins}/{n} = {wins/n:.3f}", flush=True)
