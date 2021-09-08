# project2-openmpi
This series of small program uses the multi-core processing capabilities of OpenMPI to calculate a Mandelbrot set.

A Mandelbrot set is a mathematical set of complex numbers that arises from a particular function (https://en.wikipedia.org/wiki/Mandelbrot_set). When plotted with colours, it shows a nice pattern (see 'mandelbrot.png'). The reason that we are using this set for this application is because this function has dense regions and not-so-dense regions. This means that there is extreme variability in the amount of calculation required for each of the points. Hence, multi-core processing is a method that can be used to minimise the calculation time. There are 3 different methods of distributing calculation jobs that have been put on discplay in 3 different c files: static partitioning, cyclic partitioning, and master-worker parallelism. A description of each type of partitioning is shown in their associated c file. 

## Installation & Use
In order to be able to run each program, you must be; operating on a system with multiple cores in the cpu, have OpenMPI installed, and have a valid gcc compiler. 

