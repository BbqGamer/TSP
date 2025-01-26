#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define PROBLEM_SIZE 200
#define SOLUTION_SIZE 100
#define PERTURB_NUM_AFFECTED 10
#define NUM_THREADS 16
#define UNSELECTED_SIZE (PROBLEM_SIZE - SOLUTION_SIZE)
#define NUM_ITERATIONS 20
#define NODE uint8_t
#define DIST int
#define SCORE int32_t

DIST D[PROBLEM_SIZE * PROBLEM_SIZE];
NODE solutions[NUM_THREADS][SOLUTION_SIZE]; 
time_t start;

int16_t mod(int16_t a, int16_t b)
{
    int16_t r = a % b;
    return r < 0 ? r + b : r;
}

void random_starting_solution(NODE *solution, NODE *unselected) {
    NODE i, nodes[PROBLEM_SIZE];
    
    for (i = 0; i < PROBLEM_SIZE; i++) {
        nodes[i] = i;
    }

    for (i = 0; i < PROBLEM_SIZE; i++) {
        int j = mod(rand(), PROBLEM_SIZE);
        int tmp = nodes[i];
        nodes[i] = nodes[j];
        nodes[j] = tmp;
    }

    for(i = 0; i < SOLUTION_SIZE; i++) {
        solution[i] = nodes[i];
    }

    for(i = 0; i < UNSELECTED_SIZE; i++) {
        unselected[i] = nodes[SOLUTION_SIZE + i];
    }
}

SCORE score(NODE *solution, DIST *D) {
    SCORE score = 0;
    NODE i;
    for (NODE i = 0; i < SOLUTION_SIZE - 1; i++) {
        score += D[solution[i] * PROBLEM_SIZE + solution[i + 1]];
    }
    score += D[solution[SOLUTION_SIZE - 1] * PROBLEM_SIZE + solution[0]];
    return score;
}

bool is_valid(NODE *solution) {
    bool present[PROBLEM_SIZE];
    NODE i;
    memset(present, false, sizeof(present));
    for (i = 0; i < SOLUTION_SIZE; i++) {
        if (present[solution[i]] || solution[i] < 0 || solution[i] >= PROBLEM_SIZE)
            return false;
        present[solution[i]] = true;
    }
    return true;
}

void intra_edge_exchange(NODE l, NODE r, NODE* solution) {
  l = l + 1;
  l %= SOLUTION_SIZE;
  while (l != r) {
    NODE tmp = solution[l];
    solution[l] = solution[r];
    solution[r] = tmp;

    l = mod(l + 1, SOLUTION_SIZE);
    if (l == r) {
      break;
    }
    r = mod(r - 1, SOLUTION_SIZE);
  }
}

SCORE local_search(NODE* solution, NODE* unselected, DIST* D) {
    SCORE delta, best_delta, best_score = score(solution, D);
    bool intra_best, improved = true;
    NODE bestl, bestr, aprev, a, anext, b, bnext, i, j;
    NODE rolled_indices[SOLUTION_SIZE];
    while (improved) {
        improved = false;
        best_delta = 0;

        /* intra-route edge exchange */
        for (i = 0; i < SOLUTION_SIZE; i++) {
            for (j = 0; j < SOLUTION_SIZE; j++) {
                if (abs(i - j) < 2 || abs(i - j) > SOLUTION_SIZE - 2) {
                    continue;
                }
                a = solution[i];
                b = solution[j];
                anext = solution[mod(i + 1, SOLUTION_SIZE)];
                bnext = solution[mod(j + 1, SOLUTION_SIZE)];
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

        /* inter-route node exchange */
        for (i = 0; i < SOLUTION_SIZE; i++) {
            for (j = 0;  j < UNSELECTED_SIZE; j++) {
                a = solution[i];
                aprev = solution[mod(i - 1, SOLUTION_SIZE)];
                anext = solution[mod(i + 1, SOLUTION_SIZE)];
                b = unselected[j];
                delta = D[aprev * PROBLEM_SIZE + b] + D[b * PROBLEM_SIZE + anext]
                      - D[aprev * PROBLEM_SIZE + a] - D[a * PROBLEM_SIZE + anext];
                if (delta < best_delta) {
                    best_delta = delta;
                    intra_best = false;
                    bestl = i;
                    bestr = j;
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

            int real_score = score(solution, D);
        }
    }
    return best_score;
}

void perturb(NODE* solution) {
    NODE l, r, tmp, start = rand() % SOLUTION_SIZE;
    for(uint8_t i = 0; i < PERTURB_NUM_AFFECTED; i++) {
        l = start;
        r = mod(rand(), SOLUTION_SIZE);

        if (abs(l - r) < 2 || abs(l - r) > SOLUTION_SIZE - 2) {
            i -= 1;
            continue;
        }

        intra_edge_exchange(l, r, solution);
    }
}

SCORE ILS(NODE* solution, DIST* D, float timelimit) {
    NODE unselected[UNSELECTED_SIZE];
    random_starting_solution(solution, unselected);

    int i = 0;
    SCORE best_score = score(solution, D);

    NODE cursol[SOLUTION_SIZE];
    NODE curuns[UNSELECTED_SIZE];

    time_t curtime;
    time(&curtime);
    while (difftime(curtime, start) < timelimit) {
        memcpy(cursol, solution, SOLUTION_SIZE);
        memcpy(curuns, unselected, UNSELECTED_SIZE);

        perturb(cursol);

        int sol_score = local_search(cursol, curuns, D);

        if (sol_score < best_score) {
            best_score = sol_score;
            memcpy(solution, cursol, SOLUTION_SIZE);
            memcpy(unselected, curuns, UNSELECTED_SIZE);
        }

        if (i % 100 == 0) {
            time(&curtime);
        }
        i += 1;
    }

    return best_score;
}

void *msilsrun(void *solptr) {
    NODE* solution = (NODE*) solptr;
    ILS(solution, D, 2.5);
    return NULL;
}

SCORE MSILS(NODE* solution, DIST* D, float timelimit) {
    pthread_t threads[NUM_THREADS];
    time(&start);
    
    int i;
    for(i = 0 ; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, msilsrun, (void *)solutions[i]);
    }

    for(i = 0 ; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    SCORE cur_score, best_score = score(solutions[0], D);
    memcpy(solution, solutions[0], SOLUTION_SIZE);
    printf("thread results: ");
    for(i = 1; i < NUM_THREADS; i++) {
        cur_score = score(solutions[i], D);
        printf("%d ", cur_score);
        if (cur_score < best_score) {
            best_score = cur_score;
            memcpy(solution, solutions[i], SOLUTION_SIZE);
        }
    }
    printf("\n");

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
    NODE i, j;
    for (i = 0; i < PROBLEM_SIZE; i++) {
        if (!fgets(line, sizeof(line), file)) {
            break;
        }

        if (line[0] == '\n') {
            i--;
            continue;
        }

        sscanf(line, "%d;%d;%d", &points[i][0], &points[i][1], &weights[i]);
    }

    for (i = 0; i < PROBLEM_SIZE; i++) {
        for (j = 0; j < PROBLEM_SIZE; j++) {
            /* euclidian distance */
            D[i * PROBLEM_SIZE + j] = floor(sqrt(
                pow(points[i][0] - points[j][0], 2) +
                pow(points[i][1] - points[j][1], 2)) + 0.5
            ) + weights[j];
        }
    }

    NODE solution[SOLUTION_SIZE];

    for(i = 0; i < NUM_ITERATIONS; i++) {
        SCORE best_score = MSILS(solution, D, 2.5);

        printf("%d\n", best_score);
        
        char prefix[4] = {"sol"};
        char filename[100];
        sprintf(filename, "%s%d", prefix, i);

        FILE *output = fopen(filename, "w");
        for (int i = 0; i < SOLUTION_SIZE; i++) {
            fprintf(output, "%d\n", solution[i]);
        }
        fclose(output);
    }
}
