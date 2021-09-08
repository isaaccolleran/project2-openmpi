
#define MAXITER 1000
#define N	8000
//#define DEBUG
//#define SAVE
#define TIME

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>

int check_info_array(int *A, int size);
//void fill_array(float *x_chunk, int start_idx, int chunk_size);

/* ----------------------------------------------------------------*/
// PROGRAM static_partitioning
// AUTHOR Isaac Colleran
// ABOUT this program calculates the Mandelbrot set using multi-cores
//       distributed using a master-worker technique. This means that
//       one core is designated as the 'master'. The master's job is 
//       to send jobs off to each of the other cores ('workers'), 
//       receive the completed work, and send a new job back. This
//       essentially means that there is a constant communication line
//       between the master and its workers.
/* ----------------------------------------------------------------*/


void main(int argc, char *argv[]) 
{
    int    chunk_size, rank, ncores, curr_idx, work_done, i_core;
    int    green, blue, loop, source, i, j, k, max_iter, i_iter, copy_idx;
    int    tag = 0;
    int    *info;
    float  complex z, kappa;
    float  *x_chunk, *x;
    #ifdef TIME
        double start_time, calc_time, wait_time, comm_time, total_time;
    #endif
    FILE   *fp;
    MPI_Status *status;
    MPI_Request *request;


    if (argc == 2)
    {
        chunk_size = atoi(argv[1]);
    }
    else
    {
        chunk_size = 1000; // default chunk_size
    }
    x_chunk = (float *) malloc(chunk_size * sizeof(float));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncores);

/* ------------------------------------ MASTER ------------------------------------ */
    if (rank == 0)
    {
        // contains the starting indexes of each of the cores currently being processed
        info = (int *) malloc((ncores-1) * sizeof(int));

        x = (float *) malloc(N*N*sizeof(float));

        curr_idx = 0;
        work_done = 0; // work not yet finished

        // Handing out initial work
        for (i_core = 1; i_core<ncores; i_core++)
        {
            info[i_core-1] = curr_idx;

            // Need to send the starting index and the number of loop iterations
            MPI_Send(&curr_idx, 1, MPI_INT, i_core, 0, MPI_COMM_WORLD);

            curr_idx += chunk_size;
        }
        #ifdef DEBUG
            printf("{ROOT}: All initial jobs have been sent!\n");
            fflush(stdout);
        #endif


        i_iter = 0;
        max_iter = N*N / chunk_size + 1; // basically ceiling function

        // Now that all jobs have been handed out, need to continuously receive and send more work
        for (i_iter=0; i_iter<max_iter; i_iter++)
        {

            // "info" array contains all the starting indexes for the chunks currently being processed.
            // Once all of the jobs have been assigned, these indexes in the array will start to be set to
            // -1 suggesting that there is no work left for that particular core to do.

            // Receives the data from any source
            MPI_Recv(x_chunk, chunk_size, MPI_FLOAT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, status);

            source = status->MPI_SOURCE; // works out which source the data is coming from
            copy_idx = info[source-1]; // the first index that the source was processing

            #ifdef DEBUG
                printf("{ROOT}: RECEIVED FROM RANK %d\n", source);
                fflush(stdout);
            #endif

            // Now must send data back to that same source
            
            if (info[source-1] != -1)
            {
                info[source-1] = curr_idx;
                #ifdef DEBUG
                    printf("{ROOT}: SENDING %d TO %d\n", curr_idx, i_core);
                    fflush(stdout);
                #endif
                MPI_Send(&curr_idx, 1, MPI_INT, source, tag, MPI_COMM_WORLD);
                // if curr_idx is -1 it means that all chunks have been allocated and sent to cores.

                // In the case where info[source-1] != -1, but curr_idx == -1:
                //      --> All chunks have been allocated but core number source still doesn't 
                //          know that. 
                //      --> Therefore, we must send a -1 to that core to let it know that there 
                //          is nothing left for it to do and to not throw up another receive


                // Master knows which indexes each cores are calculating 
                // So, now we need to take that data and put it into the main malloc
                for (loop=copy_idx; loop < copy_idx+chunk_size && loop < N*N; loop++)
                {
                    x[loop] = x_chunk[loop-copy_idx];
                }

                if (curr_idx != -1)
                {
                    curr_idx += chunk_size;
                }
                
                // Sends the next chunk and updates the starting index of the next chunk to send
            }
            // else nothing left to send

            if (curr_idx > N*N) 
            {
                curr_idx = -1;
            }
        }

        free(info);
        free(x_chunk);

        #ifdef SAVE
            #ifdef DEBUG
                printf("{ROOT}: Writing mandelbrot.ppm\n");
                fflush(stdout);
            #endif

            fp = fopen ("mandelbrot.ppm", "w");
            fprintf (fp, "P3\n%4d %4d\n255\n", N, N);
        
            for (loop=0; loop<N*N; loop++)
            { 
                if (x[loop]<0.5)
                {
                    green = (int) (2*x[loop]*255);
                    fprintf (fp, "%3d\n%3d\n%3d\n", 255-green,green,0);
                } 
                else
                {
                    blue = (int) (2*x[loop]*255-255);
                    fprintf (fp, "%3d\n%3d\n%3d\n", 0,255-blue,blue);
                }
            }
            fclose(fp);
        #endif

        free(x);

        #ifdef DEBUG
            printf("{ROOT}: DONE\n");
            fflush(stdout);
        #endif

    }
/* ------------------------------------ MASTER ------------------------------------ */


/* ------------------------------------ WORKER ------------------------------------ */
    else
    {
        do
        {
            // Receive the work
            MPI_Recv(&curr_idx, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, status);

            // if the index received is -1, it means that there is no more work for
            // the core to do and it shouldn't throw up another receive. Therefore, 
            // there is also nothing to send and hence will not enter the if statement
            // below and the loop will also be terminated.
            
            #ifdef DEBUG
                printf("{%d}: RECEIVED %d FROM %d\n", rank, curr_idx, 0);
                fflush(stdout);
            #endif

            if (curr_idx != -1)
            {
                // Do the work

                for (loop=curr_idx; loop<(curr_idx+chunk_size) && loop<N*N; loop++)
                {
                    i=loop/N;
                    j=loop%N;

                    z = kappa = (4.0*(i-N/2))/N + (4.0*(j-N/2))/N * I;

                    k=1;
                    while ((cabs(z)<=2) && (k++<MAXITER)) 
                        z= z*z + kappa;

                    x_chunk[loop-curr_idx]= log((float)k) / log((float)MAXITER);
                }

                // Send the results
                MPI_Send(x_chunk, chunk_size, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
            } 
            
        } while (curr_idx != -1);

        #ifdef DEBUG
            printf("{%d}: DONE\n", rank);
            fflush(stdout);
        #endif
    }   
/* ------------------------------------ WORKER ------------------------------------ */
  
  MPI_Finalize();
}

// This function is now redundant
int check_info_array(int *A, int size)
{
    int ii;
    int target = -1;
    int is_same = 1;

    ii = 0;
    do
    {
        if (A[ii] != A[ii+1]) {is_same = 0;}
    } while (is_same==1 && ++ii<size);
    
    while (A[ii]==A[++ii])
    
    /* BITWISE IMPLEMENTATION (NOT WORKING)
    for (ii=0; ii<size; ii++) target |= A[ii];
    // Uses a bitwise OR assignment operation
    // If, after this for loop, the value of sum is still 0 
    //      -> A (array) contains only -1

    if (target == -1) {is_empty = 1;}
    // Will return 1 if the array contains only -1s */

    return is_same;
}

/*
void fill_array(float *x_chunk, int start_idx, int chunk_size)
{
    int i, j, k, loop;
    float complex z, kappa;

    for (loop=start_idx; loop<(start_idx+chunk_size) && loop<N*N; loop++)
    {
        i=loop/N;
        j=loop%N;

        z = kappa = (4.0*(i-N/2))/N + (4.0*(j-N/2))/N * I;
        
        k=1;
        while ((cabs(z)<=2) && (k++<MAXITER)) 
            z= z*z + kappa;
        
        x_chunk[loop-start_idx]= log((float)k) / log((float)MAXITER);
    }

} */

