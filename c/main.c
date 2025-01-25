#include <stdio.h>
#include <unistd.h>
#include <math.h>

#define PROBLEM_SIZE 200

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

    int D[PROBLEM_SIZE * PROBLEM_SIZE];
    for (int i = 0; i < PROBLEM_SIZE; i++) {
        for (int j = 0; j < PROBLEM_SIZE; j++) {
            /* euclidian distance */
            D[i * PROBLEM_SIZE + j] = floor(sqrt(
                pow(points[i][0] - points[j][0], 2) +
                pow(points[i][1] - points[j][1], 2)) + 0.5
            ) + weights[j];
        }
    }

    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", D[i * PROBLEM_SIZE + j]);
        }
    }
}