#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"

//
//  benchmarking program
//

int main(int argc, char **argv) {
    int navg, nabsavg = 0, numthreads, tid ,index;
    double dmin, absmin = 1.0, davg, absavg = 0.0;

    if (find_option(argc, argv, "-h") >= 0) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t*) malloc(n * sizeof (particle_t));
    set_size(n);
    double size = get_size();
    double cutoff = get_cutoff();
    init_particles(n, particles);
    double binLength = 2 * cutoff;
    int numBins = ceil(size / binLength);
    vector< vector < particle_t > > bins(numBins * numBins);

    for(int i = 0; i < numBins*numBins; i++)
        bins[i].reserve(100);
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    //
    //double bintime = 0;
    //
#pragma omp parallel private(dmin, tid, index) 
    {
        numthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
        for (int step = 0; step < NSTEPS; step++) {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            //
            //generate bins
            //
#pragma omp for schedule(guided)
            for (int i = 0; i < numBins * numBins; i++){
                 bins[i].clear();
            }
#pragma omp for
            //error may occur here
          // #pragma omp single 
            //{
                //bintime -= read_timer();
                for (int i = 0; i < n; i++) {                    
                    index = particles[i].binIndex;  
                    if(tid==index%numthreads)
                         bins[index].push_back(particles[i]);
                }
                //bintime += read_timer();
           // }
            
          /*  for (int i = 0; i < n; i++) {                    
                    index = particles[i].binIndex;  
                    if(tid==index%numthreads)
                         bins[index].push_back(particles[i]);
                }*/
            
            //
            //  compute forces for bins
            //
            int currRowIndex;
            int upRowIndex;
            int lowRowIndex;
            //
            //first bin row special case (i=0))
            //
            currRowIndex = 0;
            //
            //1st column special case
            //
#pragma omp single 
            {
                //compare with self
#pragma omp task
                self_bins_apply_force(bins[0], &dmin, &davg, &navg);
                //compare with right
#pragma omp task
                bins_apply_force(bins[0], bins[1], &dmin, &davg, &navg);
                //compare low
#pragma omp task
                bins_apply_force(bins[0], bins[numBins], &dmin, &davg, &navg);
                //compare with low-right
#pragma omp task
                bins_apply_force(bins[0], bins[numBins + 1], &dmin, &davg, &navg);
            }
            //
            //jth bin column for j=1 to numBins-2
            //
#pragma omp for reduction(+:davg) reduction(+:navg) schedule(guided) 
            for (int j = 1; j <= numBins - 2; j++) {
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
            }
            //
            //last bin column (j = numBins-1)
            //
#pragma omp single 
            {
                //compare with left
#pragma omp task
                bins_apply_force(bins[numBins - 1], bins[numBins - 2], &dmin, &davg, &navg);
                //compare with self
#pragma omp task
                self_bins_apply_force(bins[numBins - 1], &dmin, &davg, &navg);
                //compare with low-left
#pragma omp task
                bins_apply_force(bins[numBins - 1], bins[numBins + numBins - 2], &dmin, &davg, &navg);
                //compare with low
#pragma omp task
                bins_apply_force(bins[numBins - 1], bins[numBins + numBins - 1], &dmin, &davg, &navg);
            }
            //
            //ith bin row for i = 1 to numBins-2
            //
#pragma omp for reduction (+:navg) reduction(+:davg) private(currRowIndex,upRowIndex,lowRowIndex) schedule(guided) 
            for (int i = 1; i <= numBins - 2; i++) {
                //
                //1st column special case (j=0)
                //
                currRowIndex = i*numBins;
                upRowIndex = (i - 1) * numBins;
                lowRowIndex = (i + 1) * numBins;
                //compare with up
                bins_apply_force(bins[currRowIndex], bins[upRowIndex], &dmin, &davg, &navg);
                //compare with up-right
                bins_apply_force(bins[currRowIndex], bins[upRowIndex + 1], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currRowIndex], &dmin, &davg, &navg);
                //compare with right
                bins_apply_force(bins[currRowIndex], bins[currRowIndex + 1], &dmin, &davg, &navg);
                //compare with low
                bins_apply_force(bins[currRowIndex], bins[lowRowIndex], &dmin, &davg, &navg);
                //compare with low-right
                bins_apply_force(bins[currRowIndex], bins[lowRowIndex + 1], &dmin, &davg, &navg);

                //
                //jth column for j=1 to numBins-2
                //
                for (int j = 1; j <= numBins - 2; j++) {
                    //compare with up-left
                    bins_apply_force(bins[currRowIndex + j], bins[upRowIndex + j - 1], &dmin, &davg, &navg);
                    //compare with up
                    bins_apply_force(bins[currRowIndex + j], bins[upRowIndex + j], &dmin, &davg, &navg);
                    //compare with up-right
                    bins_apply_force(bins[currRowIndex + j], bins[upRowIndex + j + 1], &dmin, &davg, &navg);
                    //compare with left
                    bins_apply_force(bins[currRowIndex + j], bins[currRowIndex + j - 1], &dmin, &davg, &navg);
                    //compare with self
                    self_bins_apply_force(bins[currRowIndex + j], &dmin, &davg, &navg);
                    //compare with right
                    bins_apply_force(bins[currRowIndex + j], bins[currRowIndex + j + 1], &dmin, &davg, &navg);
                    //compare with low-left
                    bins_apply_force(bins[currRowIndex + j], bins[lowRowIndex + j - 1], &dmin, &davg, &navg);
                    //compare with low
                    bins_apply_force(bins[currRowIndex + j], bins[lowRowIndex + j], &dmin, &davg, &navg);
                    //compare with low-right
                    bins_apply_force(bins[currRowIndex + j], bins[lowRowIndex + j + 1], &dmin, &davg, &navg);
                }

                //
                //numBins-1 (last column) special case
                //
                //compare with up-left
                bins_apply_force(bins[currRowIndex + numBins - 1], bins[upRowIndex + numBins - 2], &dmin, &davg, &navg);
                //compare with up
                bins_apply_force(bins[currRowIndex + numBins - 1], bins[upRowIndex + numBins - 1], &dmin, &davg, &navg);
                //compare with left
                bins_apply_force(bins[currRowIndex + numBins - 1], bins[currRowIndex + numBins - 2], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currRowIndex + numBins - 1], &dmin, &davg, &navg);
                //compare with low-left
                bins_apply_force(bins[currRowIndex + numBins - 1], bins[lowRowIndex + numBins - 2], &dmin, &davg, &navg);
                //compare with low
                bins_apply_force(bins[currRowIndex + numBins - 1], bins[lowRowIndex + numBins - 1], &dmin, &davg, &navg);
            }

            //
            //numBins-1 (last row)
            //
            currRowIndex = (numBins - 1) * numBins;
            upRowIndex = (numBins - 2) * numBins;
            //1st column special case (j=0)
#pragma omp single 
            {
                //compare up
#pragma omp task
                bins_apply_force(bins[currRowIndex], bins[upRowIndex], &dmin, &davg, &navg);
                //compare up-right
#pragma omp task
                bins_apply_force(bins[currRowIndex], bins[upRowIndex + 1], &dmin, &davg, &navg);
                //compare self
#pragma omp task
                self_bins_apply_force(bins[currRowIndex], &dmin, &davg, &navg);
                //compare right
#pragma omp task
                bins_apply_force(bins[currRowIndex], bins[currRowIndex + 1], &dmin, &davg, &navg);
            }
            //
            //jth column for j=1 to numBins-2
            //
#pragma omp for reduction (+:navg) reduction(+:davg) schedule(guided) 
            for (int j = 1; j <= numBins - 2; j++) {
                //compare with up-left
                bins_apply_force(bins[currRowIndex + j], bins[upRowIndex + j - 1], &dmin, &davg, &navg);
                //compare with up
                bins_apply_force(bins[currRowIndex + j], bins[upRowIndex + j], &dmin, &davg, &navg);
                //compare with up-right
                bins_apply_force(bins[currRowIndex + j], bins[upRowIndex + j + 1], &dmin, &davg, &navg);
                //compare with left
                bins_apply_force(bins[currRowIndex + j], bins[currRowIndex + j - 1], &dmin, &davg, &navg);
                //compare with self
                self_bins_apply_force(bins[currRowIndex + j], &dmin, &davg, &navg);
                //compare with right
                bins_apply_force(bins[currRowIndex + j], bins[currRowIndex + j + 1], &dmin, &davg, &navg);
            }
            //
            //numBins-1 (last column) special case
            //
#pragma omp single private(currRowIndex)
            {
                currRowIndex = numBins * numBins - 1;
                //compare with up-left
#pragma omp task
                bins_apply_force(bins[currRowIndex], bins[upRowIndex + numBins - 2], &dmin, &davg, &navg);
                //compare with up
#pragma omp task
                bins_apply_force(bins[currRowIndex], bins[upRowIndex + numBins - 1], &dmin, &davg, &navg);
                //compare with left
#pragma omp task
                bins_apply_force(bins[currRowIndex], bins[currRowIndex - 1], &dmin, &davg, &navg);
                //compare with self
#pragma omp task
                self_bins_apply_force(bins[currRowIndex], &dmin, &davg, &navg);
            }
#pragma omp taskwait
            //
            // update and move particles
            //
#pragma omp for schedule(guided)
            for (int i = 0; i < bins.size(); i++)
                for (int j = 0; j < bins[i].size(); j++) {
                    index = bins[i][j].index;
                    particles[index] = bins[i][j];
                    pmove(particles[index]);
                }

            if (find_option(argc, argv, "-no") == -1) {
                //
                //  compute statistical data
                //
#pragma omp master
                if (navg) {
                    absavg += davg / navg;
                    nabsavg++;
                }

#pragma omp critical
                if (dmin < absmin) absmin = dmin;

                //
                //  save if necessary
                //
#pragma omp master
                if (fsave && (step % SAVEFREQ) == 0)
                    save(fsave, n, particles);
            }
        }
    }
    simulation_time = read_timer() - simulation_time;

    printf("n = %d,threads = %d, simulation time = %g seconds", n, numthreads, simulation_time);

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
        fprintf(fsum, "%d %d %g\n", n, numthreads, simulation_time);

    //
    // Clearing space
    //
    if (fsum)
        fclose(fsum);

    free(particles);
    if (fsave)
        fclose(fsave);

    //printf("bin time %g seconds\n", bintime);

    return 0;
}
