#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// #define VERBOSE 1

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set* frontier, vertex_set* new_frontier, int* distances){
    for (int i=0; i<frontier->count; i++) {
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

void parallel_top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int *distances){
    int *nodes_to_add = (int *)malloc(sizeof(int) * g->num_nodes);
    int count = 0;

    #pragma omp for schedule(dynamic, 256)
    for (int i = 0; i < frontier->count; i++){
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];


        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++){
            int outgoing = g->outgoing_edges[neighbor];
            if (distances[outgoing] == NOT_VISITED_MARKER && __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)){
                nodes_to_add[count++] = outgoing;
            }
        }
    }

    if (count > 0){
        int offset = __sync_fetch_and_add(&new_frontier->count, count);
        memcpy(new_frontier->vertices + offset, nodes_to_add, sizeof(int) * count);
    }
}

void parallel_bottom_up_step(Graph g, vertex_set *frontier, vertex_set *new_frontier, int* distances, bool* in_frontier){
    int *nodes_to_add = (int *)malloc(sizeof(int) * g->num_nodes);
    int count = 0;

    #pragma omp for schedule(dynamic, 2048)
    for (int node = 0; node < g->num_nodes; node++){
        if (distances[node] != NOT_VISITED_MARKER)
            continue;

        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++){
            int incoming = g->incoming_edges[neighbor];
            if (in_frontier[incoming]){
                nodes_to_add[count++] = node;
                distances[node] = distances[incoming] + 1;
                break;
            }
        }
    }

    if (count > 0){
        int offset = __sync_fetch_and_add(&new_frontier->count, count);
        memcpy(new_frontier->vertices + offset, nodes_to_add, sizeof(int) * count);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    #pragma omp parallel
    {
        while (frontier->count != 0){

            #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
            #endif
            
            #pragma omp single
            vertex_set_clear(new_frontier);

            parallel_top_down_step(graph, frontier, new_frontier, sol->distances);

            #ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            #pragma omp single
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
            #endif

            #pragma omp single
            {
                // swap pointers
                vertex_set *tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *in_frontier = (bool *)malloc(sizeof(bool) * graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    memset(in_frontier, false, sizeof(bool) * graph->num_nodes);
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    in_frontier[ROOT_NODE_ID] = true;

    #pragma omp parallel
    {
        while (frontier->count != 0){

            #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
            #endif

            #pragma omp single
            vertex_set_clear(new_frontier);
            parallel_bottom_up_step(graph, frontier, new_frontier, sol->distances, in_frontier);

            #ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            #pragma omp single
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
            #endif

            #pragma omp single
            {
                vertex_set *tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
            }

            // #pragma omp for schedule(static, 16)
            // for(int i=0; i<graph->num_nodes; i++){
            //     in_frontier[i] = false;
            // }
            #pragma omp for schedule(static, 64)
            for(int i=0; i<frontier->count; i++){
                in_frontier[frontier->vertices[i]] = true;
            }
        }
    }

    delete[] in_frontier;
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *in_frontier = (bool *)malloc(sizeof(bool) * graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    memset(sol->distances, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    memset(in_frontier, false, sizeof(bool) * graph->num_nodes);
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    in_frontier[ROOT_NODE_ID] = true;

    #pragma omp parallel
    {
        while (frontier->count != 0){

            #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
            #endif

            #pragma omp single
            vertex_set_clear(new_frontier);
            if((double)new_frontier->count < ((double)graph->num_nodes) * 0.03)
                parallel_top_down_step(graph, frontier, new_frontier, sol->distances);
            else
                parallel_bottom_up_step(graph, frontier, new_frontier, sol->distances, in_frontier);

            #ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            #pragma omp single
            printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
            #endif

            #pragma omp single
            {
                vertex_set *tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
            }

            // #pragma omp for schedule(static, 16)
            // for(int i=0; i<graph->num_nodes; i++){
            //     in_frontier[i] = false;
            // }
            #pragma omp for schedule(static, 64)
            for(int i=0; i<frontier->count; i++){
                in_frontier[frontier->vertices[i]] = true;
            }
        }
    }

    delete[] in_frontier;
}