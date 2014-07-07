#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <upc.h>

//
// auxiliary functions
//
inline int max( int a, int b ) { return a > b ? a : b; }
inline int min( int a, int b ) { return a < b ? a : b; }
double read_timer( )
{
    static int initialized = 0;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = 1;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}


//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    int i;
    for(i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

typedef shared [] int *sintptr;
shared sintptr directory[THREADS];
shared sintptr weights[THREADS];
shared sintptr values[THREADS];

//
//  solvers
//
int build_table( int n, int W, shared int * numrows)
{
    int i,j,k;
           
    long dirsize = (W+1)*numrows[MYTHREAD]*sizeof(int);
    int * mydir = (int *) malloc(dirsize);

    sintptr v = values[MYTHREAD];
    sintptr w = weights[MYTHREAD];

    int colperblock = (W+1+10*THREADS-1)/(10*THREADS);
    int numcolblock = (W+1+colperblock-1)/colperblock;    
    
    int myfirstrow = 0;
    for(i = 0; i < MYTHREAD; i++)
        myfirstrow += numrows[i];

    shared int * ready = (shared int *) upc_all_alloc((THREADS-1)*numcolblock,sizeof(int));    
    
    if(MYTHREAD == 0){       

        for(j = 0; j < numcolblock; j++){ 
         
            int colblockoffset = j*colperblock;

            int wi = w[0];
            int vi = v[0];
            int knumiter = min(wi-colblockoffset, colperblock);
            for(k=0; k < knumiter; k++)
                mydir[colblockoffset+k] = 0;
            int next = max(wi-colblockoffset,0);
            knumiter = min(colperblock,W+1-colblockoffset); 
            for(k = next; k < knumiter; k++)
                mydir[colblockoffset+k]=vi; 
            for(i = 1; i < numrows[MYTHREAD]; i++){
                wi = w[i];
                vi = v[i];
                int offset = i*(W+1) + colblockoffset;
                int upoffset = (i-1)*(W+1) + colblockoffset;
                knumiter = min(wi-colblockoffset, colperblock);
                for(k = 0; k < knumiter; k++)
                    mydir[offset+k] = mydir[upoffset+k];  
                int next = max(wi-colblockoffset,0);
                knumiter = min(colperblock,W+1-colblockoffset); 
                for(k = next; k < knumiter; k++)
                    mydir[offset+k] = max(mydir[upoffset+k],mydir[upoffset+k-wi]+vi);
            } 
            upc_memput(directory[MYTHREAD]+(numrows[MYTHREAD]-1)*(W+1),mydir+(numrows[MYTHREAD]-1)*(W+1),(W+1)*sizeof(int));
            ready[j]=1;
        }
    }

    else if(MYTHREAD != THREADS-1){        
        
        int * toprow = (int *) malloc((W+1)*sizeof(int));
        sintptr updir = directory[MYTHREAD-1];
        int upnumrows = numrows[MYTHREAD-1];  

        for(j = 0; j < numcolblock; j++){

            int colblockoffset = j*colperblock;
            int wi = w[0];
            int vi = v[0]; 
            
            while(!ready[(MYTHREAD-1)*numcolblock+j]){} 

            upc_memget(toprow, &updir[(upnumrows-1)*(W+1)],(W+1)*sizeof(int));
            
            int knumiter = min(wi-colblockoffset,colperblock);
            for(k = 0; k < knumiter; k++)
                mydir[colblockoffset+k] = toprow[colblockoffset+k];
            int next = max(wi-colblockoffset,0);
            knumiter = min(colperblock,W+1-colblockoffset);
            for(k = next; k < knumiter; k++)
                mydir[colblockoffset+k] = max(toprow[colblockoffset+k],toprow[colblockoffset+k-wi]+vi);

            for(i = 1; i < numrows[MYTHREAD]; i++){
                wi = w[i];
                vi = v[i];
                int offset = i*(W+1) + colblockoffset;
                int upoffset = (i-1)*(W+1) + colblockoffset;
                knumiter = min(wi-colblockoffset,colperblock);
                for(k = 0; k < knumiter; k++)
                    mydir[offset+k] = mydir[upoffset+k];
                next = max(wi-j*colperblock,0);
                knumiter = min(colperblock,W+1-colblockoffset);
                for(k = next; k < knumiter; k++)
                    mydir[offset+k] = max(mydir[upoffset+k],mydir[upoffset+k-wi]+vi);         
            }          
            upc_memput(directory[MYTHREAD]+(numrows[MYTHREAD]-1)*(W+1),mydir+(numrows[MYTHREAD]-1)*(W+1),(W+1)*sizeof(int));
            ready[MYTHREAD*numcolblock+j]=1;
        }
        
        free(toprow);
        
    }

    else {
        int * toprow = (int *) malloc((W+1)*sizeof(int));
        sintptr updir = directory[MYTHREAD-1];
        int upnumrows = numrows[MYTHREAD-1];

        for(j = 0; j < numcolblock; j++){

            int colblockoffset = j*colperblock;
            int wi = w[0];
            int vi = v[0]; 

            while(!ready[(MYTHREAD-1)*numcolblock+j]){} 
           
            upc_memget(toprow, &updir[(upnumrows-1)*(W+1)],(W+1)*sizeof(int));
            
            int knumiter = min(wi-colblockoffset,colperblock);
            for(k = 0; k < knumiter; k++)
                mydir[colblockoffset+k] = toprow[colblockoffset+k];
            int next = max(wi-colblockoffset,0);
            knumiter = min(colperblock,W+1-colblockoffset);
            for(k = next; k < knumiter; k++)
                mydir[colblockoffset+k] = max(toprow[colblockoffset+k],toprow[colblockoffset+k-wi]+vi);

            for(i = 1; i < numrows[MYTHREAD]; i++){
                wi = w[i];
                vi = v[i];
                int offset = i*(W+1) + colblockoffset;
                int upoffset = (i-1)*(W+1) + colblockoffset;
                knumiter = min(wi-colblockoffset,colperblock);
                for(k = 0; k < knumiter; k++)
                    mydir[offset+k] = mydir[upoffset+k];
                next = max(wi-j*colperblock,0);
                knumiter = min(colperblock,W+1-colblockoffset);
                for(k = next; k < knumiter; k++)
                    mydir[offset+k] = max(mydir[upoffset+k],mydir[upoffset+k-wi]+vi);         
            } 
        }
        
        free(toprow);
    }

    upc_memput(directory[MYTHREAD], mydir, dirsize);
    free(mydir);

    upc_barrier;

    if(MYTHREAD==0){
        upc_free(ready);
    }

    sintptr lastdir = directory[THREADS-1];
    int lastnumrows = n/THREADS;
    if( THREADS-1 < n%THREADS)
        lastnumrows++;
    return lastdir[(lastnumrows-1)*(W+1)+W];
}

void backtrack( int n, int W, shared int *w, shared int *used, shared int * numrows )
{
    if(MYTHREAD != 0) return;

    int i,j;

    int * T = (int *) malloc((W+1)*n*sizeof(int)); 
 
    int row = 0;
    for(i = 0; i < THREADS; i++){
        upc_memget(T+row*(W+1),directory[i],numrows[i]*(W+1)*sizeof(int));
        row+=numrows[i];
    }   

    i = n*(W+1) - 1;
    for( j = n-1; j > 0; j-- )
    {
        used[j] = T[i] != T[i-W-1];
        i -= W+1 + (used[j] ? w[j] : 0 );
    }
    used[0] = T[i] != 0;

    free(T);
}

//
//  serial solver to check correctness
//
int solve_serial( int nitems, int cap, shared int *w, shared int *v )
{
    int i, j, best, *allocated, *T, wj, vj;

    //alloc local resources
    T = allocated = malloc( nitems*(cap+1)*sizeof(int) );
    if( !allocated )
    {
        fprintf( stderr, "Failed to allocate memory" );
        upc_global_exit( -1 );
    }

    //build_table locally
    wj = w[0];
    vj = v[0];
    for( i = 0;  i <  wj;  i++ ) T[i] = 0;
    for( i = wj; i <= cap; i++ ) T[i] = vj;
    for( j = 1; j < nitems; j++ )
    {
        wj = w[j];
        vj = v[j];
        for( i = 0;  i <  wj;  i++ ) T[i+cap+1] = T[i];
        for( i = wj; i <= cap; i++ ) T[i+cap+1] = max( T[i], T[i-wj]+vj );
        T += cap+1;
    }
    best = T[cap];

    //free resources
    free( allocated );

    return best;
}

//
//  benchmarking program
//
int main( int argc, char** argv )
{
    int i, best_value, best_value_serial, total_weight, nused, total_value;
    double seconds;
    shared int *used;
    shared int *numrows;
    shared int *weight;
    shared int *value;

	if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-c <int> to set the knapsack capacity\n" );
        printf( "-n <nitems> to specify the number of items\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        return 0;
    }
	
    //these constants have little effect on runtime
    int max_value  = 1000;
    int max_weight = 1000;

    //reading in problem size
    int capacity   = read_int( argc, argv, "-c", 999 );
    int nitems     = read_int( argc, argv, "-n", 5000 );
	
    srand48( (unsigned int)time(NULL) + MYTHREAD );
    
    //allocate distributed arrays, use cyclic distribution
    used   = (shared int *) upc_all_alloc( nitems, sizeof(int) );
    numrows = (shared int *) upc_all_alloc(THREADS,sizeof(int));
    weight = (shared int *) upc_all_alloc( nitems, sizeof(int) );
    value  = (shared int *) upc_all_alloc( nitems, sizeof(int) );
    if( !weight || !value || !used || !numrows)
    {
        fprintf( stderr, "Failed to allocate memory" );
        upc_global_exit( -1 );
    }

    int localrows = nitems/THREADS;
    if( MYTHREAD < nitems%THREADS)
        localrows++;
    numrows[MYTHREAD] = localrows;

    //allocate total directories
    directory[MYTHREAD] = (sintptr) upc_alloc((capacity+1)*localrows*sizeof(int));
    if( !directory[MYTHREAD] )
    {
        fprintf( stderr, "Failed to allocate directories memory" );
        upc_global_exit( -1 );
    }
    
    upc_barrier;

    int myfirstrow = 0;
    for(i = 0; i < MYTHREAD; i++)
        myfirstrow += numrows[i]; 
    
    //allocate total, weight, value directories
    weights[MYTHREAD] = upc_alloc(localrows*sizeof(int)); 
    values[MYTHREAD] = upc_alloc(localrows*sizeof(int));
    if( !weights[MYTHREAD] || !values[MYTHREAD] )
    {
        fprintf( stderr, "Failed to allocate weights or values memory" );
        upc_global_exit( -1 );
    }

    // init
    max_weight = min( max_weight, capacity );//don't generate items that don't fit into bag
    sintptr myweights = weights[MYTHREAD];
    sintptr myvalues = values[MYTHREAD];
    for(i = 0; i < localrows; i++)
    {
        int w = 1 + (lrand48()%max_weight);
        myweights[i] = w;
        weight[myfirstrow+i] = w;

        int v = 1 + (lrand48()%max_value);
        myvalues[i] = v;
        value[myfirstrow+i] = v;
    }

    upc_barrier;

    // time the solution
    seconds = read_timer( );

    best_value = build_table(nitems, capacity, numrows);
       
    if(MYTHREAD==0)
    printf("time building table = %g\n",read_timer()-seconds);

    backtrack(nitems, capacity, weight, used, numrows);
    
    seconds = read_timer( ) - seconds;

    // check the result
    if( MYTHREAD == 0 )
    {
        printf( "%d items, capacity: %d, time: %g\n", nitems, capacity, seconds );

        best_value_serial = solve_serial( nitems, capacity, weight, value );

        total_weight = nused = total_value = 0;
        for( i = 0; i < nitems; i++ )
            if( used[i] )
            {
                nused++;
                total_weight += weight[i];
                total_value += value[i];
            }

        printf( "%d items used, value %d, weight %d\n", nused, total_value, total_weight );

        if( best_value != best_value_serial || best_value != total_value || total_weight > capacity )
            printf( "WRONG SOLUTION\n" );
		
		// Doing summary data
		char *sumname = read_string( argc, argv, "-s", NULL );
		FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;
		if( fsum) 
			fprintf(fsum,"%d %d %d %g\n",nitems,capacity,THREADS,seconds);
		if( fsum )
			fclose( fsum );    
		

        //release resources
        upc_free( weight );
        upc_free( value );
        upc_free( used );
        upc_free( numrows );
    }

    upc_free(directory[MYTHREAD]);
    upc_free(weights[MYTHREAD]);
    upc_free(values[MYTHREAD]);

    return 0;
}
