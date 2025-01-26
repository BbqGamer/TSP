#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define PROBLEM_SIZE 200
#define SOLUTION_SIZE 100
#define PERTURB_NUM_AFFECTED 10
#define UNSELECTED_SIZE (PROBLEM_SIZE - SOLUTION_SIZE)
#define NUM_ITERATIONS 20
#define NODE uint8_t
#define DIST int

int abs(int x) {
    return x < 0 ? -x : x;
}

void random_starting_solution(NODE *solution, NODE *unselected) {
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

    for(int i = 0; i < UNSELECTED_SIZE; i++) {
        unselected[i] = nodes[SOLUTION_SIZE + i];
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

bool is_valid(NODE *solution) {
    bool present[PROBLEM_SIZE];
    memset(present, false, sizeof(present));
    for (int i = 0; i < SOLUTION_SIZE; i++) {
        if (present[solution[i]] || solution[i] < 0 || solution[i] >= PROBLEM_SIZE)
            return false;
        present[solution[i]] = true;
    }
    return true;
}

void intra_edge_exchange(NODE l, NODE r, NODE* solution) {
  l = (l + 1) % SOLUTION_SIZE;
  while (l != r) {
    NODE tmp = solution[l];
    solution[l] = solution[r];
    solution[r] = tmp;

    l = (l + 1) % SOLUTION_SIZE;
    if (l == r) {
      break;
    }
    r = (r - 1);
    r %= SOLUTION_SIZE;
  }
}


int local_search(NODE* solution, NODE* unselected, DIST* D) {
    int best_score = score(solution, D);
    int improved = true;
    int delta, best_delta;
    bool intra_best;
    NODE bestl, bestr;
    NODE aprev, a, anext, b, bnext;
    NODE rolled_indices[SOLUTION_SIZE];
    while (improved) {
        improved = false;
        best_delta = 0;

        /* intra-route edge exchange */
        for (int i = 0; i < SOLUTION_SIZE; i++) {
            for (int j = 0; j < SOLUTION_SIZE; j++) {
                if (abs(i - j) < 2 || abs(i - j) > SOLUTION_SIZE - 2) {
                    continue;
                }
                a = solution[i];
                b = solution[j];
                anext = solution[(i + 1) % SOLUTION_SIZE];
                bnext = solution[(j + 1) % SOLUTION_SIZE];
                delta = D[bnext * PROBLEM_SIZE + anext] + D[a * PROBLEM_SIZE + b]
                      - D[a * PROBLEM_SIZE + anext] - D[bnext * PROBLEM_SIZE + b];
                if (delta < best_delta) {
                    best_delta = delta;
                    intra_best = true;
                    bestl = i;
                    bestr = j;
                }
            }
        }

        /* inter-route edge exchange */
        for (int i = 0; i < SOLUTION_SIZE; i++) {
            for (int k = 0;  k < UNSELECTED_SIZE; k++) {
                a = solution[i];
                aprev = solution[(i - 1) % SOLUTION_SIZE];
                anext = solution[(i + 1) % SOLUTION_SIZE];
                b = unselected[k];
                delta = D[aprev * PROBLEM_SIZE + b] + D[b * PROBLEM_SIZE + anext]
                      - D[aprev * PROBLEM_SIZE + a] - D[a * PROBLEM_SIZE + anext];
                if (delta < best_delta) {
                    best_delta = delta;
                    intra_best = false;
                    bestl = i;
                    bestr = k;
                }
            }
        }

        if (best_delta < 0) {
            improved = true;
            best_score += best_delta;
            if (intra_best) {
                intra_edge_exchange(bestl, bestr, solution);
            } else {
                a = solution[bestl];
                b = unselected[bestr];
                solution[bestl] = b;
                unselected[bestr] = a;
            }
        }
    }
    return best_score;
}

void perturb(NODE* solution) {
    uint8_t start = rand() % SOLUTION_SIZE;
    uint8_t l, r;
    NODE tmp;
    for(int i = 0; i < PERTURB_NUM_AFFECTED; i++) {
        l = start;
        r = rand() % SOLUTION_SIZE;

        if (abs(l - r) < 2 || abs(l - r) > SOLUTION_SIZE - 2) {
            i -= 1;
            continue;
        }

        intra_edge_exchange(l, r, solution);
    }
}

int ILS(NODE* solution, NODE* unselected, DIST* D, float timelimit) {
    double cur = clock();
    double end = cur + timelimit * CLOCKS_PER_SEC;
    int best_score = score(solution, D);

    NODE cursol[SOLUTION_SIZE];
    NODE curuns[UNSELECTED_SIZE];
    while (cur < end) {
        memcpy(cursol, solution, SOLUTION_SIZE);
        memcpy(curuns, unselected, UNSELECTED_SIZE);

        perturb(cursol);

        int score_tofix = local_search(cursol, curuns, D);
        int sol_score = score(cursol, D);

        if (sol_score < best_score) {
            best_score = sol_score;
            memcpy(solution, cursol, SOLUTION_SIZE);
            memcpy(unselected, curuns, UNSELECTED_SIZE);
        }

        cur = clock();
    }

    return best_score;
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
    NODE unselected[UNSELECTED_SIZE];

    for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
        random_starting_solution(starting_solution, unselected);
        int best_score = ILS(starting_solution, unselected, D, 2.5);

        printf("%d\n", best_score);
    }

    FILE *output = fopen("bestsol", "w");
    for (int i = 0; i < SOLUTION_SIZE; i++) {
        fprintf(output, "%d\n", starting_solution[i]);
    }
    fclose(output);
}
