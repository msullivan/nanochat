"""Print all timing-relevant fields from a single wandb run, to figure out
which one the dashboard's Runtime column actually displays."""
import argparse
import json
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_path", help="entity/project/run_id")
    args = p.parse_args()

    api = wandb.Api()
    r = api.run(args.run_path)

    print("=== summary fields ===")
    for k, v in dict(r.summary).items():
        if "time" in k.lower() or "runtime" in k.lower() or k.startswith("_") or "duration" in k.lower():
            print(f"  {k}: {v}")

    print("\n=== top-level attrs ===")
    interesting = ["createdAt", "heartbeatAt", "updatedAt", "stoppedAt", "endTime",
                   "duration", "runtime", "state", "lastUpdated"]
    for k in interesting:
        if hasattr(r, k):
            print(f"  attr {k}: {getattr(r, k)}")

    print("\n=== raw _attrs (GraphQL fields) ===")
    for k, v in (r._attrs or {}).items():
        if any(t in k.lower() for t in ("time", "duration", "runtime", "heartbeat", "stop", "end", "created", "update")):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
