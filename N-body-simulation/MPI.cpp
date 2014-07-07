#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <math.h>
#include <vector>
//
//  benchmarking program
//
using namespace std;

int main(int argc, char **argv) {
    int navg, nabsavg = 0, count, currBinIndex, i, j;
    double dmin, absmin = 1.0, davg, absavg = 0.0;
    double rdavg, rdmin;
    int rnavg;
    particle_t p;

    //
    //  process command line parameters
    //
    if (find_option(argc, argv, "-h") >= 0) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t*) malloc(n * sizeof (particle_t));

    vector < particle_t > updParticles;
    updParticles.reserve(n / n_proc);

    //TODO test
    vector < particle_t > lowParticles;
    lowParticles.reserve(n / n_proc);

    vector < particle_t > bufParticles(n);

    vector< int > rcvCount(n_proc);
    vector< int > rcvDisplacements(n_proc);

    vector< int > countCount;
    countCount.resize(n_proc, 1);
    vector< int > countDisplacements;
    for (int i = 0; i < n_proc; i++)
        countDisplacements.push_back(i);


    /* MPI_Datatype PARTICLE;
      int blockcounts[] = {6, 2};
      MPI_Aint extent;
      MPI_Type_extent(MPI_DOUBLE, &extent);
      MPI_Aint offsets[] = {0, 6 * extent};
      MPI_Datatype oldtypes[] = {MPI_DOUBLE, MPI_INT};
      MPI_Type_struct(2, blockcounts, offsets, oldtypes, &PARTICLE);
      MPI_Type_commit(&PARTICLE); */


    MPI_Datatype PARTICLE;
    MPI_Type_contiguous(8, MPI_DOUBLE, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size(n);
    double size = get_size();
    double cutoff = get_cutoff();
    double binLength = 2 * cutoff;
    int numBins = (int) ceil(size / binLength);

    int pB = (int) ceil(numBins / n_proc);
    int firstBin = rank * pB*numBins;
    int lastBin = (rank + 1) * pB * numBins - 1;
    if (lastBin >= numBins * numBins)
        lastBin = numBins * numBins - 1;

    vector< vector < particle_t > > bins(numBins * numBins);
    for (int i = 0; i < numBins * numBins; i++)
        bins[i].reserve(100);

    if (rank == 0)
        init_particles(n, particles);

    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();

    double timer = 0;

    for (int step = 0; step < NSTEPS; step++) {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if (find_option(argc, argv, "-no") == -1)
            if (fsave && (step % SAVEFREQ) == 0)
                save(fsave, n, particles);

        //
        //generate bins
        // !
        for (int i = 0; i < numBins * numBins; i++)
            bins[i].clear();
        int index;
        for (int i = 0; i < n; i++) {
            bins[(int) particles[i].binIndex].push_back(particles[i]);
        }

        //
        //  compute forces for bins
        // !
        currBinIndex = firstBin;
        //handle first bin row -------------------------------------------------
        //
        //1st column special case (j=0))
        //            
        if (currBinIndex == 0) {
            //compare with self
            self_bins_apply_force(bins[0], &dmin, &davg, &navg);
            //compare with right
            bins_apply_force(bins[0], bins[1], &dmin, &davg, &navg);
            //compare low
            bins_apply_force(bins[0], bins[numBins], &dmin, &davg, &navg);
            //compare with low-right
            bins_apply_force(bins[0], bins[numBins + 1], &dmin, &davg, &navg);
            //printf("(%d) handled bins[%d] (0,0) \n", rank, currBinIndex);
            currBinIndex++;
        }
        //
        //jth bin column for j=1 to numBins-2
        //
        while (currBinIndex < numBins - 1) {
            j = currBinIndex;
            //compare with left
            bins_apply_force(bins[j], bins[j - 1], &dmin, &davg, &navg);
            //compare with self
            self_bins_apply_force(bins[j], &dmin, &davg, &navg);
            //compare with right
            bins_apply_force(bins[j], bins[j + 1], &dmin, &davg, &navg);
            //compare with low-left
            bins_apply_force(bins[j], bins[numBins + j - 1], &dmin, &davg, &navg);
            //compare with low
            bins_apply_force(bins[j], bins[numBins + j], &dmin, &davg, &navg);
            //compare with low-right    
            bins_apply_force(bins[j], bins[numBins + j + 1], &dmin, &davg, &navg);
            //printf("(%d) handled bins[%d] (0,j)\n", rank, currBinIndex);
            currBinIndex++;
        }
        //
        //last bin column (j = numBins-1)
        //
        if (currBinIndex == numBins - 1) {
            //compare with left
            bins_apply_force(bins[numBins - 1], bins[numBins - 2], &dmin, &davg, &navg);
            //compare with self
            self_bins_apply_force(bins[numBins - 1], &dmin, &davg, &navg);
            //compare with low-left
            bins_apply_force(bins[numBins - 1], bins[numBins + numBins - 2], &dmin, &davg, &navg);
            //compare with low
            bins_apply_force(bins[numBins - 1], bins[numBins + numBins - 1], &dmin, &davg, &navg);
            //printf("(%d) handled bins[%d] (0,last)\n", rank, currBinIndex);
            currBinIndex++;
        }

        //
        //ith bin rows for i = 1 to numBins-2
        //
        while (currBinIndex < (numBins - 1) * numBins && currBinIndex <= lastBin && currBinIndex < numBins * numBins) {
            //
            //1st column special case (j=0)
            //
            if (currBinIndex % numBins == 0) {
                //compare with up
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins], &dmin, &davg, &navg);
                //compare with up-right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins + 1], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currBinIndex], &dmin, &davg, &navg);
                //compare with right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + 1], &dmin, &davg, &navg);
                //compare with low
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins], &dmin, &davg, &navg);
                //compare with low-right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins + 1], &dmin, &davg, &navg);
            }//
                //jth column for j=1 to numBins-2
                //
            else if (currBinIndex % numBins != (numBins - 1)) {
                //compare with up-left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins - 1], &dmin, &davg, &navg);
                //compare with up
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins], &dmin, &davg, &navg);
                //compare with up-right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins + 1], &dmin, &davg, &navg);
                //compare with left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - 1], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currBinIndex], &dmin, &davg, &navg);
                //compare with right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + 1], &dmin, &davg, &navg);
                //compare with low-left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins - 1], &dmin, &davg, &navg);
                //compare with low
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins], &dmin, &davg, &navg);
                //compare with low-right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins + 1], &dmin, &davg, &navg);
            }//
                //numBins-1 (last column) special case
                //
            else {
                //compare with up-left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins - 1], &dmin, &davg, &navg);
                //compare with up
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins], &dmin, &davg, &navg);
                //compare with left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - 1], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currBinIndex], &dmin, &davg, &navg);
                //compare with low-left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins - 1], &dmin, &davg, &navg);
                //compare with low
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins], &dmin, &davg, &navg);
            }
            //printf("(%d) handled bins[%d] (i,j)\n", rank, currBinIndex);
            currBinIndex++;
        }

        //
        //numBins-1 (last row)
        //
        while (currBinIndex >= (numBins - 1) * numBins && currBinIndex < numBins * numBins && currBinIndex <= lastBin) {
            //
            //1st column special case (j=0)
            //
            if (currBinIndex % numBins == 0) {
                //compare up
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins], &dmin, &davg, &navg);
                //compare up-right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins + 1], &dmin, &davg, &navg);
                //compare self
                self_bins_apply_force(bins[currBinIndex], &dmin, &davg, &navg);
                //compare right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + 1], &dmin, &davg, &navg);
            }//
                //jth column for j=1 to numBins-2
                //
            else if (currBinIndex % numBins != (numBins - 1)) {
                //compare with up-left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins - 1], &dmin, &davg, &navg);
                //compare with up
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins], &dmin, &davg, &navg);
                //compare with up-right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - numBins + 1], &dmin, &davg, &navg);
                //compare with left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - 1], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currBinIndex], &dmin, &davg, &navg);
                //compare with right
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + 1], &dmin, &davg, &navg);
            }//
                //numBins-1 (last column) special case
                //
            else {
                //compare with up-left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins - 1], &dmin, &davg, &navg);
                //compare with up
                bins_apply_force(bins[currBinIndex], bins[currBinIndex + numBins], &dmin, &davg, &navg);
                //compare with left
                bins_apply_force(bins[currBinIndex], bins[currBinIndex - 1], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currBinIndex], &dmin, &davg, &navg);
            }
            //printf("(%d) handled bins[%d] (last, j)\n", rank, currBinIndex);
            currBinIndex++;
        }


        /*currBinIndex = firstBin;
        updParticles.clear();
        while (currBinIndex <= lastBin && currBinIndex < numBins*numBins) {
            for (int i = 0; i < bins[currBinIndex].size(); i++) {
                p = bins[currBinIndex][i];
                pmove(p);
                if((p.binIndex <= (firstBin+numBins-1) && firstBin!=0) || (p.binIndex >= (lastBin-numBins+1) && lastBin < (numBins-1)*numBins))
                    updParticles.push_back(p);
                else
                  particles[(int)p.index] = p;                
            }
            currBinIndex++;
        }

        //all processes will receive all corresponding particles !
        count = updParticles.size();
        MPI_Allgatherv(&count, 1, MPI_INT, &rcvCount[0], &countCount[0], &countDisplacements[0], MPI_INT, MPI_COMM_WORLD);
        int bufSize = rcvCount[0];
        rcvDisplacements[0]=0;
        for (int i = 1; i < n_proc; i++){
            rcvDisplacements[i] = rcvDisplacements[i - 1] + rcvCount[i - 1];
            bufSize+=rcvCount[i];
        }
        bufParticles.resize(bufSize);
        MPI_Allgatherv(&updParticles[0], count, PARTICLE, &bufParticles[0], &rcvCount[0], &rcvDisplacements[0], PARTICLE, MPI_COMM_WORLD);*/


        //
        //test------------------------------------------------------------------
        //
        currBinIndex = firstBin;
        updParticles.clear();
        lowParticles.clear();
        while (currBinIndex <= lastBin && currBinIndex < numBins * numBins) {
            for (int i = 0; i < bins[currBinIndex].size(); i++) {
                p = bins[currBinIndex][i];
                pmove(p);
                if (p.binIndex <= (firstBin + numBins - 1) && firstBin != 0)
                    updParticles.push_back(p);
                else if (p.binIndex >= (lastBin - numBins + 1) && lastBin < (numBins - 1) * numBins)
                    lowParticles.push_back(p);
                else
                    particles[(int) p.index] = p;
            }
            currBinIndex++;
        }


        //even send down first
        if (rank % 2 == 0) {
            if (rank + 1 < n_proc) {
                //send count to low
                count = lowParticles.size();
                MPI_Send(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                //send data to low
                MPI_Send(&lowParticles[0], count, PARTICLE, rank + 1, 1, MPI_COMM_WORLD);

                //receive count from low
                MPI_Recv(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bufParticles.resize(count);
                //receive data from low
                MPI_Recv(&bufParticles[0], count, PARTICLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; i++)
                    particles[(int) bufParticles[i].index] = bufParticles[i];
            }

            if (rank != 0) {
                //send count to up
                count = updParticles.size();
                MPI_Send(&count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                //send data to up
                MPI_Send(&updParticles[0], count, PARTICLE, rank - 1, 1, MPI_COMM_WORLD);

                //receive count from up
                MPI_Recv(&count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bufParticles.resize(count);
                //receive data from up
                MPI_Recv(&bufParticles[0], count, PARTICLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; i++)
                particles[(int) bufParticles[i].index] = bufParticles[i];
            }
        }
            //odd receive up first
        else {
            //receive count from up
            MPI_Recv(&count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            bufParticles.resize(count);
            //receive data from up
            MPI_Recv(&bufParticles[0], count, PARTICLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++)
                particles[(int) bufParticles[i].index] = bufParticles[i];

            //send count to up
            count = updParticles.size();
            MPI_Send(&count, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
            //send data to up
            MPI_Send(&updParticles[0], count, PARTICLE, rank - 1, 1, MPI_COMM_WORLD);

            if (rank + 1 < n_proc) {
                //receive count from low
                MPI_Recv(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bufParticles.resize(count);
                //receive data from low
                MPI_Recv(&bufParticles[0], count, PARTICLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; i++)
                    particles[(int) bufParticles[i].index] = bufParticles[i];
                
                //send count to low
                 count = lowParticles.size();
                MPI_Send(&count, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                //send data to low
                MPI_Send(&lowParticles[0], count, PARTICLE, rank + 1, 1, MPI_COMM_WORLD);
            }
        }

        //
        //test------------------------------------------------------------------
        //

        /*//reorganize particles based on their index !
        for (int i = 0; i < bufSize; i++)
            particles[(int)bufParticles[i].index] = bufParticles[i];*/

        if (find_option(argc, argv, "-no") == -1) {

            MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);


            if (rank == 0) {
                //
                // Computing statistical data
                //
                if (rnavg) {
                    absavg += rdavg / rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin) absmin = rdmin;
            }
        }
    }
    simulation_time = read_timer() - simulation_time;

    if (rank == 0) {
        printf("n = %d, simulation time = %g seconds", n, simulation_time);

        if (find_option(argc, argv, "-no") == -1) {
            if (nabsavg) absavg /= nabsavg;
            // 
            //  -the minimum distance absmin between 2 particles during the run of the simulation
            //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
            //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
            //
            //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
            //
            printf(", absmin = %lf, absavg = %lf", absmin, absavg);
            if (absmin < 0.4) printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
            if (absavg < 0.8) printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
        printf("\n");

        //  
        // Printing summary data
        //  
        if (fsum)
            fprintf(fsum, "%d %d %g\n", n, n_proc, simulation_time);
    }

    //
    //  release resources
    //
    if (fsum)
        fclose(fsum);
    free(particles);
    if (fsave)
        fclose(fsave);

    MPI_Finalize();

    return 0;
}
