import sys
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')

if __name__ == "__main__":
    from deephokm.training.gauntlet_workers import run_team_gauntlet_shards, OpponentSpec
    for path, name in [("checkpoints/bc_256d.zip", "bc_clone"), ("checkpoints/distill_search.zip", "distill_hand1")]:
        wins = run_team_gauntlet_shards(path, OpponentSpec(kind="greedy"),
                                        n_games=200, n_workers=16, seed=99)
        print(f"{name}: {wins}/200 = {wins/200:.3f}", flush=True)
