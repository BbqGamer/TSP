#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define PROBLEM_SIZE 200
#define SOLUTION_SIZE 100
#define NODE uint8_t
#define DIST int

void random_starting_solution(NODE *solution) {
    NODE nodes[PROBLEM_SIZE];
    for (int i = 0; i < PROBLEM_SIZE; i++) {
        nodes[i] = i;
    }

    for (int i = 0; i < PROBLEM_SIZE; i++) {
        int j = rand() % PROBLEM_SIZE;
        int tmp = nodes[i];
        nodes[i] = nodes[j];
        nodes[j] = tmp;
    }

    for(int i = 0; i < SOLUTION_SIZE; i++) {
        solution[i] = nodes[i];
    }
}

int score(NODE *solution, DIST *D) {
    int score = 0;
    for (int i = 0; i < SOLUTION_SIZE - 1; i++) {
        score += D[solution[i] * PROBLEM_SIZE + solution[i + 1]];
    }
    score += D[solution[SOLUTION_SIZE - 1] * PROBLEM_SIZE + solution[0]];
    return score;
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        fprintf(stderr, "Usage %s <filename>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", argv[1]);
        return 1;
    }

    int points[PROBLEM_SIZE][2];
    int weights[PROBLEM_SIZE];

    char line[128];
    for (int i = 0; i < PROBLEM_SIZE; i++) {
        if (!fgets(line, sizeof(line), file)) {
            break;
        }

        if (line[0] == '\n') {
            i--;
            continue;
        }

        sscanf(line, "%d;%d;%d", &points[i][0], &points[i][1], &weights[i]);
    }

    for (int i = 0; i < PROBLEM_SIZE; i++) {
        printf("%d %d %d\n", points[i][0], points[i][1], weights[i]);
    }

    DIST D[PROBLEM_SIZE * PROBLEM_SIZE];
    for (int i = 0; i < PROBLEM_SIZE; i++) {
        for (int j = 0; j < PROBLEM_SIZE; j++) {
            /* euclidian distance */
            D[i * PROBLEM_SIZE + j] = floor(sqrt(
                pow(points[i][0] - points[j][0], 2) +
                pow(points[i][1] - points[j][1], 2)) + 0.5
            ) + weights[j];
        }
    }

    NODE starting_solution[SOLUTION_SIZE];
    random_starting_solution(starting_solution);

    for (int i = 0; i < SOLUTION_SIZE; i++) {
        printf("%d ", starting_solution[i]);
    }

    printf("\n");
    printf("%d", score(starting_solution, D));
}
