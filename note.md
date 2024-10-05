# How OpenMP works
```cpp
#pragma omp parallel
{
    // Initialization
    /*
        Before First Construct(BFC):
            Run only by master thread, it initializes the shared variables that will be used in parallel region.
    */
    #pragma omp for
    for (...) {
        // Distributed tasks are running in different thread
    }
    /*
        After First Construct(AFC):
            Run by every thread concurrently, pay attention to the race condition!
    */
    #pragma omp sections
    {
        #pragma omp section
        {
            // Run by one thread
        }
        #pragma omp section
        {
            // Different code, run by another thread
        }
    }

    // AFC
    
    while (...) {
        #pragma omp task
        {
            // Run asychronously by available threads
        }
    }
    #pragma taskwait // wait for asyc threads to complete

    #pragma cirtical
    {
        // critical section
    }

    #pragma single 
    {
        // Executed once, unspecific thread
    }

    /*
        Still an AFC. Run by all threads
    */
}
```