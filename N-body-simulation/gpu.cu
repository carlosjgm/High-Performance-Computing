#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

extern double size;
//
//  benchmarking program
//

__global__ void clear_binSize(int * d_binSize, int numBins) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= numBins * numBins) return;

    d_binSize[tid] = 0;

}

__global__ void generate_bins(particle_t * particles, particle_t * * bins, int * binSize, int n, int pitch) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) return;

    int currBinIndex = particles[tid].binIndex;
    int index = atomicAdd(&binSize[currBinIndex], 1);

    bins[currBinIndex * pitch + index] = &particles[tid];

}

__device__ void apply_force_gpu(particle_t * particle, particle_t neighbor) {
    double dx = neighbor.x - particle->x;
    double dy = neighbor.y - particle->y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    //r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r*min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle->ax += coef * dx;
    particle->ay += coef * dy;
}

__device__ void bin_apply_force_gpu(particle_t * particle, particle_t * * bin, int length) {
   for (int i = 0; i < length; i++)
       apply_force_gpu(particle, *bin[i]);
    //particle->newIndex = bin[i]->index + 10; //ok
}

__global__ void compute_forces_gpu(particle_t * particles, particle_t * * bins, int * binSize, int n, int numBins, int pitch) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) return;

    particle_t * currParticle = &particles[tid];
    int currBinIndex = currParticle->binIndex;

    //handle first bin row -------------------------------------------------
    //
    //1st column special case (j=0))
    //            
    if (currBinIndex == 0) {
        //compare with self
        bin_apply_force_gpu(currParticle, &bins[0], binSize[0]);
        //compare with right
        bin_apply_force_gpu(currParticle, &bins[pitch], binSize[1]);
        //compare low
        bin_apply_force_gpu(currParticle, &bins[numBins * pitch], binSize[numBins]);
        //compare with low-right
        bin_apply_force_gpu(currParticle, &bins[(numBins + 1) * pitch], binSize[numBins + 1]);
    }//
        //jth bin column for j=1 to numBins-2
        //
    else if (currBinIndex < numBins - 1) {
        //compare with left
        bin_apply_force_gpu(currParticle, &bins[(currBinIndex - 1) * pitch], binSize[currBinIndex - 1]);
        //compare with self
        bin_apply_force_gpu(currParticle, &bins[currBinIndex * pitch], binSize[currBinIndex]);
        //compare with right
        bin_apply_force_gpu(currParticle, &bins[(currBinIndex + 1) * pitch], binSize[currBinIndex + 1]);
        //compare with low-left
        bin_apply_force_gpu(currParticle, &bins[(numBins + currBinIndex - 1) * pitch], binSize[numBins + currBinIndex - 1]);
        //compare with low
        bin_apply_force_gpu(currParticle, &bins[(numBins + currBinIndex) * pitch], binSize[numBins + currBinIndex]);
        //compare with low-right    
        bin_apply_force_gpu(currParticle, &bins[(numBins + currBinIndex + 1) * pitch], binSize[numBins + currBinIndex + 1]);
    }//
        //last bin column (j = numBins-1)
        //
    else if (currBinIndex == numBins - 1) {
        //compare with left
        bin_apply_force_gpu(currParticle, &bins[(numBins - 2) * pitch], binSize[numBins - 2]);
        //compare with self
        bin_apply_force_gpu(currParticle, &bins[(numBins - 1) * pitch], binSize[numBins - 1]);
        //compare with low-left
        bin_apply_force_gpu(currParticle, &bins[(numBins + numBins - 2) * pitch], binSize[numBins + numBins - 2]);
        //compare with low
        bin_apply_force_gpu(currParticle, &bins[(numBins + numBins - 1) * pitch], binSize[numBins + numBins - 1]);
    }//
        //ith bin rows for i = 1 to numBins-2
        //
  else if (currBinIndex < (numBins - 1) * numBins) {
        //
        //1st column special case (j=0)
        //
        if (currBinIndex % numBins == 0) {
            //compare with up
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins) * pitch], binSize[currBinIndex - numBins]);
            //compare with up-right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins + 1) * pitch], binSize[currBinIndex - numBins + 1]);
            //compare with self
            bin_apply_force_gpu(currParticle, &bins[currBinIndex * pitch], binSize[currBinIndex]);
            //compare with right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + 1) * pitch], binSize[currBinIndex + 1]);
            //compare with low
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins) * pitch], binSize[currBinIndex + numBins]);
            //compare with low-right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins + 1) * pitch], binSize[currBinIndex + numBins + 1]);
        }//
            //jth column for j=1 to numBins-2
            //
        else if (currBinIndex % numBins != (numBins - 1)) {
            //compare with up-left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins - 1) * pitch], binSize[currBinIndex - numBins - 1]);
            //compare with up
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins) * pitch], binSize[currBinIndex - numBins]);
            //compare with up-right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins + 1) * pitch], binSize[currBinIndex - numBins + 1]);
            //compare with left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - 1) * pitch], binSize[currBinIndex - 1]);
            //compare with self
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex) * pitch], binSize[currBinIndex]);
            //compare with right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + 1) * pitch], binSize[currBinIndex + 1]);
            //compare with low-left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins - 1) * pitch], binSize[currBinIndex + numBins - 1]);
            //compare with low
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins) * pitch], binSize[currBinIndex + numBins]);
            //compare with low-right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins + 1) * pitch], binSize[currBinIndex + numBins + 1]);
        }//
            //numBins-1 (last column) special case
            //
        else {
            //compare with up-left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins - 1) * pitch], binSize[currBinIndex - numBins - 1]);
            //compare with up
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins) * pitch], binSize[currBinIndex - numBins]);
            //compare with left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - 1) * pitch], binSize[currBinIndex - 1]);
            //compare with self
            bin_apply_force_gpu(currParticle, &bins[currBinIndex * pitch], binSize[currBinIndex]);
            //compare with low-left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins - 1) * pitch], binSize[currBinIndex + numBins - 1]);
            //compare with low
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins) * pitch], binSize[currBinIndex + numBins]);
        }
    }//
        //numBins-1 (last row)
        //
     else if (currBinIndex >= (numBins - 1) * numBins && currBinIndex < numBins * numBins) {
        //
        //1st column special case (j=0)
        //
        if (currBinIndex % numBins == 0) {
            //compare up
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins) * pitch], binSize[currBinIndex - numBins]);
            //compare up-right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins + 1) * pitch], binSize[currBinIndex - numBins + 1]);
            //compare self
            bin_apply_force_gpu(currParticle, &bins[currBinIndex * pitch], binSize[currBinIndex]);
            //compare right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + 1) * pitch], binSize[currBinIndex + 1]);
        }//
            //jth column for j=1 to numBins-2
            //
        else if (currBinIndex % numBins != (numBins - 1)) {
            //compare with up-left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins - 1) * pitch], binSize[currBinIndex - numBins - 1]);
            //compare with up
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins) * pitch], binSize[currBinIndex - numBins]);
            //compare with up-right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - numBins + 1) * pitch], binSize[currBinIndex - numBins + 1]);
            //compare with left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - 1) * pitch], binSize[currBinIndex - 1]);
            //compare with self
            bin_apply_force_gpu(currParticle, &bins[currBinIndex * pitch], binSize[currBinIndex]);
            //compare with right
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + 1) * pitch], binSize[currBinIndex + 1]);
        }//
            //numBins-1 (last column) special case
            //
        else {
            //compare with up-left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins - 1) * pitch], binSize[currBinIndex + numBins - 1]);
            //compare with up
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex + numBins) * pitch], binSize[currBinIndex + numBins]);
            //compare with left
            bin_apply_force_gpu(currParticle, &bins[(currBinIndex - 1) * pitch], binSize[currBinIndex - 1]);
            //compare with self
            bin_apply_force_gpu(currParticle, &bins[currBinIndex * pitch], binSize[currBinIndex]);
        }
    }
}

__global__ void move_gpu(particle_t * particles, int n, double size, int numBins, double binLength) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }

    p->ax = p->ay = 0;
    p->binIndex = ((int) (p->x / binLength)) + ((int) (p->y / binLength)) * numBins;
}

int main(int argc, char **argv) {
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if (find_option(argc, argv, "-h") >= 0) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify the summary output file name\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);

    set_size(n);
    double size = get_size();
    double binLength = 2 * cutoff;
    int numBins = ceil(size / binLength);

    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;
    setbuf(stdout, NULL);

    particle_t *particles = (particle_t*) malloc(n * sizeof (particle_t));
    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof (particle_t));

    particle_t * * d_bins;
    int perBin = 10;
    cudaMalloc((void **) &d_bins, perBin * numBins * numBins * sizeof (particle_t *));

    int * d_binSize;
    cudaMalloc((void **) &d_binSize, numBins * numBins * sizeof (int));

    init_particles(n, particles);

    cudaThreadSynchronize();
    double copy_time = read_timer();

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof (particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer() - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer();
    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    int binBlks = (numBins * numBins + NUM_THREADS - 1) / NUM_THREADS;
    for (int step = 0; step < NSTEPS; step++) {

        //
        // clear bins
        //
        clear_binSize << <binBlks, NUM_THREADS >> > (d_binSize, numBins);

        //
        //GENERATE BIN
        //
        generate_bins << < blks, NUM_THREADS >> > (d_particles, d_bins, d_binSize, n, perBin);
         
        //
        //  compute forces
        //
        compute_forces_gpu << <blks, NUM_THREADS >> > (d_particles, d_bins, d_binSize, n, numBins, perBin);
        
        //
        //  move particles
        //        
        move_gpu << < blks, NUM_THREADS >> > (d_particles, n, size, numBins, binLength);
        
        //
        //  save if necessary
        //
        if (fsave && (step % SAVEFREQ) == 0) {
            // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof (particle_t), cudaMemcpyDeviceToHost);
            save(fsave, n, particles);
        }
    }
    cudaThreadSynchronize();

    simulation_time = read_timer() - simulation_time;

    printf("CPU-GPU copy time = %g seconds\n", copy_time);
    printf("n = %d, simulation time = %g seconds\n", n, simulation_time);

    if (fsum)
        fprintf(fsum, "%d %lf \n", n, simulation_time);

    if (fsum)
        fclose(fsum);
    free(particles);
    cudaFree(d_particles);
    cudaFree(d_bins);
    cudaFree(d_binSize);
    if (fsave)
        fclose(fsave);

    return 0;
}
