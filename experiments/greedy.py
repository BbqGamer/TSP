import sys

from tsp import TSP


def main():
    if len(sys.argv) < 2:
        print("Please specify input file!")
        sys.exit(1)

    instance = TSP.from_csv(sys.argv[1])
    print(f"Loaded TSP instance")
    print(f"c: {instance.c}")


if __name__ == "__main__":
    main()
