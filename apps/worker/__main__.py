from worker.runner import main


if __name__ == "__main__":  # guard: spawned subprocesses must not re-run the CLI
    main()
