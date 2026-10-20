#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N     4096
#define ITERS 500
#define REF   4949330.8767278139
#define IDX(i,j) ((size_t)(i)*N + (j))

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* внутренние строки 1..N-2 делим на p полос, остаток раздаём первым процессам */
    int M = N - 2, base = M / p, rem = M % p;
    int nloc = base + (rank < rem ? 1 : 0);
    int first = 1 + rank * base + (rank < rem ? rank : rem);   /* глобальный номер первой своей строки */

    /* локально: nloc своих строк + 2 строки-гало (индексы 0 и nloc+1) */
    double *u    = calloc((size_t)(nloc + 2) * N, sizeof(double));
    double *unew = calloc((size_t)(nloc + 2) * N, sizeof(double));
    if (!u || !unew) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* верхняя граница = 100: для rank 0 это гало-строка 0 (в обоих массивах, не перезаписывается) */
    if (rank == 0)
        for (int j = 0; j < N; j++) u[IDX(0,j)] = unew[IDX(0,j)] = 100.0;

    int up   = (rank > 0)     ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < p - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int it = 0; it < ITERS; it++) {
        /* обмен гало: свою первую строку вверх, получить нижнее гало; и наоборот */
        MPI_Sendrecv(&u[IDX(1,0)],        N, MPI_DOUBLE, up,   0,
                     &u[IDX(nloc+1,0)],   N, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[IDX(nloc,0)],     N, MPI_DOUBLE, down, 1,
                     &u[IDX(0,0)],        N, MPI_DOUBLE, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i <= nloc; i++)
            for (int j = 1; j < N - 1; j++)
                unew[IDX(i,j)] = 0.25 * (u[IDX(i-1,j)] + u[IDX(i+1,j)]
                                       + u[IDX(i,j-1)] + u[IDX(i,j+1)]);
        double *tmp = u; u = unew; unew = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_loc = MPI_Wtime() - t0, t;
    MPI_Reduce(&t_loc, &t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* CHECK: сумма внутренних точек (вне замера) */
    double s_loc = 0.0, s = 0.0;
    for (int i = 1; i <= nloc; i++)
        for (int j = 1; j < N - 1; j++) s_loc += u[IDX(i,j)];
    MPI_Reduce(&s_loc, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double err = fabs(s - REF) / fabs(REF);
        double gbs = 16.0 * (double)M * M * ITERS / t / 1e9;
        printf("p=%d time=%.6f sum=%.10f GBs=%.1f CHECK=%s (rel.err=%.2e)\n",
               p, t, s, gbs, err < 1e-9 ? "OK" : "FAIL", err);
    }
    free(u); free(unew);
    MPI_Finalize();
    return 0;
}