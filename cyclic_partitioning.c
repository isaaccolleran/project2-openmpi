
#define MAXITER 1000
#define N	8000
//#define SAVE
#define TIME

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>

/* ----------------------------------------------------------------*/
// PROGRAM cyclic_partitioning
// AUTHOR Isaac Colleran
// ABOUT this program calculates the Mandelbrot set using multi-cores
//       partitioned cyclically. This means that each core is assigned
//       one point to calculate for every ncores.
//       For example, if we have 4 cores available on our CPU and 12
//       points to calculate:
//          core 1 - points 1, 5, 9
//          core 2 - points 2, 6, 10
//          core 3 - points 3, 7, 11
//          core 4 - points 4, 8, 12
/* ----------------------------------------------------------------*/

main(int argc, char *argv[]) 
{
    int	   i, j, k, green, blue, loop, rank, ncores;
    int    iter_per_core, start_iter, end_iter, curr_idx = 0;
    float  *x, *x_core, *x_final; 
    #ifdef TIME
        double start_time, calc_time, wait_time, comm_time, total_time;
    #endif
    FILE   *fp;
    float complex   z, kappa;

    MPI_Init(&argc, &argv); // initialise the multithreading
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets the rank of each core
    MPI_Comm_size(MPI_COMM_WORLD, &ncores); // gets the number of cores used

    if (rank==0)
    {
        x=(float *) malloc(N*N*sizeof(float)); // only the master core will save the whole array
    }

    iter_per_core = N * N / ncores; // Number of iterations per core

    x_core = (float *) malloc(iter_per_core*sizeof(float)); // array storage for each core

    for (i=0; i < iter_per_core; i++) x_core[i] = 0; // initialising array to zero

    start_iter = rank;

    #ifdef TIME
        start_time = MPI_Wtime();
    #endif

    for (loop=start_iter; loop<N*N; loop+=ncores)
    {
        i = loop/N;
        j = loop%N;

        z = kappa = (4.0*(i-N/2))/N + (4.0*(j-N/2))/N * I;
        
        k = 1;
        while ((cabs(z)<=2) && (k++<MAXITER)) 
            z = z*z + kappa;
            
        x_core[curr_idx++] = log((float)k) / log((float)MAXITER);
        // loop value is the point in the whole image, not for the local area being
        // calculated by the core. That is why we need to take away the start_iter
    }
    #ifdef TIME
        calc_time = MPI_Wtime() - start_time; 
        MPI_Barrier(MPI_COMM_WORLD);
        wait_time = MPI_Wtime() - (start_time + calc_time);
    #endif 

    MPI_Gather(x_core, iter_per_core, MPI_FLOAT, x, iter_per_core, MPI_FLOAT, 0, MPI_COMM_WORLD);

    #ifdef TIME
        comm_time = MPI_Wtime() - (start_time + calc_time + wait_time);
    #endif

/* ----------------------------------------------------------------*/
    if (rank==0)
    {
        // before writing the mandelbrot, need to make sure its all in the right order
        x_final = (float *) malloc(N*N*sizeof(float));
        curr_idx = 0;
        for (i=0; i<ncores; i++)
        {
            for (j=0; j<iter_per_core; j++)
            {
                x_final[i + j*ncores] = x[curr_idx++];
                //x_final[i*ncores + j] = x[i*iter_per_core + j];
            }

        }

        #ifdef TIME
            calc_time = MPI_Wtime() - (start_time + wait_time + comm_time);
        #endif

        #ifdef SAVE
            // writing mandelbrot
            printf("Writing mandelbrot.ppm\n");
            fflush(stdout);

            fp = fopen ("mandelbrot.ppm", "w");
            fprintf (fp, "P3\n%4d %4d\n255\n", N, N);
        
            for (loop=0; loop<N*N; loop++)
            { 
                if (x_final[loop]<0.5)
                {
                    green = (int) (2*x_final[loop]*255);
                    fprintf (fp, "%3d\n%3d\n%3d\n", 255-green,green,0);
                } 
                else
                {
                    blue = (int) (2*x_final[loop]*255-255);
                    fprintf (fp, "%3d\n%3d\n%3d\n", 0,255-blue,blue);
                }
            }
            fclose(fp);
        #endif
    }
/* ----------------------------------------------------------------*/

    if (rank==0) {free(x); free(x_final);}

    #ifdef TIME
        MPI_Barrier(MPI_COMM_WORLD);
        wait_time = MPI_Wtime() - (start_time + calc_time + comm_time);
        total_time = MPI_Wtime() - start_time;

        printf("Thread = %d: Computation time = %.3f, Communication time = %.3f, Wait time = %.3f, TOTAL = %.3f\n", rank, calc_time, comm_time, wait_time, total_time);
    #endif

    MPI_Finalize();
}
 