#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>
// #define VERBOSE

#include "../common/CycleTimer.h"
#include "../common/graph.h"

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
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    int* node_to_add = (int *)malloc(sizeof(int) * g->num_nodes);
    int cnt = 0;

    #pragma omp for schedule(guided)
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER && __sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, distances[node] + 1)) {
                // This `++` is NOT shared
                node_to_add[cnt++] = outgoing;
            }
        }
    }

    if (cnt > 0) {
        // Every thread will run this for reduction, and it seems to be atomatic.
        // I guess it is because the `cnt` is an independent variable specified in `if`
        int offset = __sync_fetch_and_add(&new_frontier->count, cnt);
        memcpy(new_frontier->vertices + offset, node_to_add, sizeof(int) * cnt);
    }

    free(node_to_add);
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
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    #pragma omp parallel
    {
        while (frontier->count != 0) {

    #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
    #endif
            #pragma omp single
            vertex_set_clear(new_frontier);

            top_down_step(graph, frontier, new_frontier, sol->distances);

    #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
    #endif

            // swap pointers
            #pragma omp single
            {
                vertex_set* tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
            }
        }
    }
}

void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances,
    std::vector<bool>& is_frontier)
{
    int cnt = 0;
    int* node_to_add = (int *)malloc(sizeof(int) * g->num_nodes);

    #pragma omp for schedule(guided)
    for (int v = 0; v < g->num_nodes; ++v) {
        if (distances[v] != NOT_VISITED_MARKER) continue;

        int op = g->incoming_starts[v];
        int ed = (v == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[v + 1];
        for (int u = op; u < ed; ++u) {
            int node = g->incoming_edges[u];
            if (is_frontier[node]) {
                // printf("I loop in %d\t", v);
                distances[v] = distances[node] + 1;
                node_to_add[cnt++] = v;

                break;
            }
        }
    }

    if (cnt > 0) {
        int offset = __sync_fetch_and_add(&new_frontier->count, cnt);
        memcpy(new_frontier->vertices + offset, node_to_add, cnt * sizeof(int));
    }

    free(node_to_add);
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

    std::vector<bool> is_frontier(graph->num_nodes, false);

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    is_frontier[ROOT_NODE_ID] = true;

    #pragma omp parallel
    {
        while (frontier->count != 0) {

    #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
    #endif
            #pragma omp single
            vertex_set_clear(new_frontier);

            bottom_up_step(graph, frontier, new_frontier, sol->distances, is_frontier);

    #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
    #endif

            // swap pointers
            #pragma omp single 
            {
                vertex_set* tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
                // std::fill(is_frontier.begin(), is_frontier.end(), false);
            }
            
            // Dynamic strategy comes with overheads of waiting for new value of the loop variable
            #pragma omp for schedule(static)
            for (int i = 0; i < frontier->count; ++i) {
                is_frontier[frontier->vertices[i]] = true;
            }
        }
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    std::vector<bool> is_frontier(graph->num_nodes, false);

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    is_frontier[ROOT_NODE_ID] = true;

    #pragma omp parallel
    {
        while (frontier->count != 0) {

    #ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
    #endif
            #pragma omp single
            vertex_set_clear(new_frontier);
            
            if (static_cast<double>(new_frontier->count) < static_cast<double>(graph->num_nodes) * 0.03)
                top_down_step(graph, frontier, new_frontier, sol->distances);
            else
                bottom_up_step(graph, frontier, new_frontier, sol->distances, is_frontier);

    #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
    #endif

            // swap pointers
            #pragma omp single 
            {
                vertex_set* tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;

                // We don't NEED to clear it, think yourself
                // std::fill(is_frontier.begin(), is_frontier.end(), false);
            }
            
            // Dynamic strategy comes with overheads of waiting for new value of the loop variable
            #pragma omp for schedule(static)
            for (int i = 0; i < frontier->count; ++i) {
                is_frontier[frontier->vertices[i]] = true;
            }
        }
    }
}
