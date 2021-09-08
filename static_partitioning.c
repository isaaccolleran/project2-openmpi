
#define MAXITER 1000
#define N	8000
#define TIME

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>

/* ----------------------------------------------------------------*/
// PROGRAM static_partitioning
// AUTHOR Isaac Colleran
// ABOUT this program calculates the Mandelbrot set using multi-cores
//       partitioned statically. This means that each core calculates
//       each of its points consecutively.
//       For example, if we have 4 cores available on our CPU and 12 
//       points to calculate:
//          core 1 - points 1, 2, 3
//          core 2 - points 4, 5, 6
//          core 3 - points 7, 8, 9
//          core 4 - points 10, 11, 12
/* ----------------------------------------------------------------*/

main(int argc, char *argv[]) {
    int	   i, j, k, green, blue, loop, rank, ncores;
    int    iter_per_core, start_iter, end_iter;
    float  *x, *temp;
    double start_time, calc_time, wait_time, comm_time;
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

    start_iter = iter_per_core * (rank);
    end_iter = iter_per_core * (rank + 1);
    // start and end points in the whole image that we loop through for this particular
    // core. 

    temp = (float *) malloc(iter_per_core*sizeof(float)); // array storage for each core

    for (i=0; i < iter_per_core; i++) temp[i] = 0;  
  
    #ifdef TIME
        start_time = MPI_Wtime();
    #endif

    for (loop=start_iter; loop<end_iter; loop++) {
	    i = loop/N;
	    j = loop%N;

	    z = kappa = (4.0*(i-N/2))/N + (4.0*(j-N/2))/N * I;
	
	    k=1;
	    while ((cabs(z)<=2) && (k++<MAXITER)) 
	        z = z*z + kappa;
	  
	    temp[loop-start_iter]= log((float)k) / log((float)MAXITER);
        // loop value is the point in the whole image, not for the local area being
        // calculated by the core. That is why we need to take away the start_iter
    }
    // after this loop need to send all the data back to the main core with MPI_Gather

    #ifdef TIME
        calc_time = MPI_Wtime() - start_time; 
        MPI_Barrier(MPI_COMM_WORLD);
        wait_time = MPI_Wtime() - (start_time + calc_time);
    #endif 

    MPI_Gather(temp, iter_per_core, MPI_FLOAT, x, iter_per_core, MPI_FLOAT, 0, MPI_COMM_WORLD); 
    // sends all back to master core
    

/* ----------------------------------------------------------------*/  
    if (rank==0)
    {

        printf("Writing mandelbrot.ppm\n");
        fp = fopen ("mandelbrot.ppm", "w");
        fprintf (fp, "P3\n%4d %4d\n255\n", N, N);
    
        for (loop=0; loop<N*N; loop++) 
	        if (x[loop]<0.5) {
	            green= (int) (2*x[loop]*255);
                    fprintf (fp, "%3d\n%3d\n%3d\n", 255-green,green,0);
	        } else {
	            blue= (int) (2*x[loop]*255-255);
                    fprintf (fp, "%3d\n%3d\n%3d\n", 0,255-blue,blue);
	        }
    
        fclose(fp);
    }
/* ----------------------------------------------------------------*/

    if (rank==0) free(x);
    
    free(temp);

    #ifdef TIME
        comm_time = MPI_Wtime() - (start_time + calc_time + wait_time);
        printf("Thread = %d: Computation time = %.3f, Communication time = %.3f, Wait time = %.3f\n", rank, calc_time, comm_time, wait_time);
    #endif

    MPI_Finalize();
}
 