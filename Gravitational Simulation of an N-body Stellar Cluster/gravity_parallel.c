#include "data.h"
#include "mt19937ar.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>

extern const double BigG, Pi, inv2Pi, LightYear, MSun, RSun, c, SecondsIn1Myear, KE_units, EscapeRadius;

int main(int argc, char** argv)
{
    /*==================================================================/
    / Main program implementing a gravitational N-body simulation in the/
    / proximity of a (non-relativistic!) black hole. The origin of the  /
    / Cartesian coordinate system is at the centre of this              /
    / black hole. X points outward to its equator. Z points out of its  /
    / North pole and Y completes a right-hand set.                      /
    /                                                                   /
    / Uses velocity-Verlet integration to track motion. Gravitational   /
    / acceleration due the black hole has been implemented for you, as  /
    / has drag due to interstellar dust. You will need to add the       /
    / gravitational interactions between the stars.                     /
    /                                                                   /
    / Simulation periodically prints the distance of the nearest object /
    / yet to get close to the black hole. Should a body reach the black /
    / hole's event horizon, its coordinates are converted to map        /
    / coordinates and written to output.dat for plotting with plot.gp   /
    / script provided.                                                  /
    /                                                                   /
    /-------------------------------------------------------------------/
    / B. Morgan - University of Warwick                                 /
    /==================================================================*/
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    double t_vv = 0.0, t_acc = 0.0, t_drag = 0.0, t_vv2 = 0.0;
    double t_book = 0.0, t_comm_pos = 0.0, t_comm_misc = 0.0;
    double t_out = 0.0;
    /*-----------------------/
    / Data/Arrays for stars  /
    /-----------------------*/
    int nStars = 1;
    int* active;    /* Star has not entered black hole (yet)  = 1 */
    double* mass;   /* Mass of each star                          */
    double* radius; /* Radius of each star                        */
    double* pos;    /* Cartesian position vectors of stars        */
    double* vel;    /* Cartesian velocity vectors of stars        */
    double* acc;    /* Cartesian acceleration vectors of stars    */

    double delta_t = SecondsIn1Myear * 0.00001;             /* Simulation timestep              */
    double BlackHoleMass = 200000 * MSun;                   /* Mass of Black Hole               */
    double BHRadius = 2.0 * BigG * BlackHoleMass / (c * c); /* Event Horizon radius             */
    double ClusterMass = 1 * MSun;                          /* Total Mass in globular cluster   */
    double ClusterRadius = 1 * LightYear;                   /* 10 light years in m              */
    double ClusterApogee = 10 * LightYear;                  /* 10 light years in m              */
    double ClusterPerigee = 10 * LightYear;                 /* 10 light years in m              */
    int Nsteps = 100000;                                    /* Number of steps to simulate      */
    double KE_init = 0.0;                                   /* Kinetic energy for initial stars */
    unsigned long seed;                                     /* Random number seed               */
    int seedtime = 0;                                       /* Use time for random seed         */

    double t1, t2; /* Timers                          */

    int istep;       /* Loop counters */
    double minRadsq; /* Squared radius of closest star */

    const int x = 0; /* For ease of reference */
    const int y = 1; /* when working with     */
    const int z = 2; /* Cartesian coords      */

    FILE* outfile; /* Output file pointer */

    /*------------------------------------/
    / Initialise random number generator  /
    /------------------------------------*/
    seed = -240882;
    if (seedtime)
    {
        seed = time(NULL);
        printf("# Using time-based random seed %ld\n", seed);
    }
    init_genrand(seed);

    /*------------------------------------------------*/
    /* Initialize simulation                          */
    /*------------------------------------------------*/
    int ra = read_args(argc, argv, &nStars, &Nsteps, &delta_t, &KE_init, &seedtime, &ClusterMass, &ClusterRadius,
                       &ClusterApogee, &ClusterPerigee);

    if (ra)
    {
        if (rank == 0)
            printf("Error Processing command line arguments\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank == 0)
    {
        printf("# Command line arguments successfully processed\n");

        printf("# BHRadius: %e\n", BHRadius / LightYear);
        printf("# delta_t: %e\n", delta_t / SecondsIn1Myear);
        printf("# ClusterMass: %e\n", ClusterMass / MSun);
        printf("# ClusterRadius: %e\n", ClusterRadius / LightYear);
        printf("# ClusterApogee: %e\n", ClusterApogee / LightYear);
        printf("# ClusterPerigee: %e\n", ClusterPerigee / LightYear);
        printf("# KE_init: %e\n", KE_init / KE_units);
    }
    /*------------------/
    / Allocate memory   /
    /------------------*/
    /* Allocate memory for mass and radius */
    mass = (double*)malloc(nStars * sizeof(double));
    radius = (double*)malloc(nStars * sizeof(double));
    if (mass == NULL)
    {
        if (rank == 0)
            printf("Error allocating mass");
        exit(EXIT_FAILURE);
    }
    if (radius == NULL)
    {
        if (rank == 0)
            printf("Error allocating radius");
        exit(EXIT_FAILURE);
    }

    /* Cartesian position, velocity and acceleration arrays */
    /* Stored as e.g. { x_0,y_0,z_0,x_1,y_1,z_1...}         */
    pos = (double*)malloc(nStars * 3 * sizeof(double));
    vel = (double*)malloc(nStars * 3 * sizeof(double));
    acc = (double*)malloc(nStars * 3 * sizeof(double));
    if (pos == NULL)
    {
        if (rank == 0)
            printf("Error allocating pos");
        exit(EXIT_FAILURE);
    }
    if (vel == NULL)
    {
        if (rank == 0)
            printf("Error allocating vel");
        exit(EXIT_FAILURE);
    }
    if (acc == NULL)
    {
        if (rank == 0)
            printf("Error allocating acc");
        exit(EXIT_FAILURE);
    }

    /* Array indicating if a star is active in the simulation */
    active = (int*)malloc(nStars * sizeof(int));
    if (active == NULL)
    {
        if (rank == 0)
            printf("Error allocating active");
        exit(EXIT_FAILURE);
    }

    /*------------------------*/
    /* Initialise the stars   */
    /*------------------------*/

    if (rank == 0)
    {
        initialiseCluster(nStars, KE_init, ClusterMass, BlackHoleMass, ClusterRadius, ClusterApogee, ClusterPerigee,
                          mass, radius, pos, vel, acc);
    }
    /* Active simulation elements */
    int nActive = nStars;
    if (rank == 0)
    {
        for (int istar = 0; istar < nStars; istar++)
        {
            active[istar] = 1;
        }
    }

    MPI_Bcast(mass, nStars, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(radius, nStars, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pos, 3 * nStars, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vel, 3 * nStars, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(active, nStars, MPI_INT, 0, MPI_COMM_WORLD);

    int base = nStars / nprocs;
    int rem = nStars % nprocs;

    int nLocal = base + (rank < rem ? 1 : 0);
    int iStart = rank * base + (rank < rem ? rank : rem);
    int iEnd = iStart + nLocal;

    /* counts/displacements for Allgatherv */
    int* counts3 = (int*)malloc(nprocs * sizeof(int));
    int* displs3 = (int*)malloc(nprocs * sizeof(int));
    int* counts1 = (int*)malloc(nprocs * sizeof(int));
    int* displs1 = (int*)malloc(nprocs * sizeof(int));

    for (int r = 0; r < nprocs; r++)
    {
        int nl = base + (r < rem ? 1 : 0);
        int st = r * base + (r < rem ? r : rem);

        counts1[r] = nl;
        displs1[r] = st;

        counts3[r] = 3 * nl;
        displs3[r] = 3 * st;
    }
    /*------------------------*/
    /* Print some diagnostics */
    /*------------------------*/
    if (rank == 0)
    {
        printf("# Generated %d stars\n", nStars);
        for (int istar = 0; istar < nStars; istar++)
        {
            int x = 3 * istar + 0;
            int y = 3 * istar + 1;
            int z = 3 * istar + 2;

            printf("# Information for star %10d\n", istar);
            printf("# -----------------------------------\n");
            printf("# Mass   : %20.12e MSun\n", mass[istar] / MSun);
            printf("# Radius : %20.12e RSun\n", radius[istar] / RSun);
            printf("# \n");
            printf("# Position :  %20.12e ,  %20.12e ,  %20.12e Ly    \n", pos[x] / LightYear, pos[y] / LightYear,
                   pos[z] / LightYear);
            printf("# Velocity :  %20.12e ,  %20.12e ,  %20.12e Ly/My \n", vel[x], vel[y], vel[z]);
            printf("# \n");
        }
    }
    /*------------------*/
    /* Open output file */
    /*------------------*/
    if (rank == 0)
    {
        outfile = fopen("output.dat", "w");
        if (outfile == NULL)
        {
            printf("Error opening output file!\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        outfile = NULL;
    }

    /*--------------------------------*/
    /* !!Start Overall timer here!!   */
    /*--------------------------------*/
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = omp_get_wtime();
    /*---------------------------*/
    /* Outer loop over timesteps */
    /*---------------------------*/

    const int s = 100; /* Output interval */
    int* eventTypeLocal = (int*)malloc(nStars * sizeof(int));
    int* eventTypeRoot = NULL;

    if (!eventTypeLocal)
    {
        fprintf(stderr, "Rank %d: failed to allocate eventTypeLocal\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0)
    {
        eventTypeRoot = (int*)malloc(nStars * sizeof(int));
        if (!eventTypeRoot)
        {
            fprintf(stderr, "Rank 0: failed to allocate eventTypeRoot\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    for (istep = 0; istep < Nsteps; istep++)
    {
        double t0;
        /* Initialise distance to closest star */
        minRadsq = DBL_MAX;
        for (int i = 0; i < nStars; i++)
        {
            eventTypeLocal[i] = -1;
        }
        /*==========================================================*/
        /* Velocity Verlet Algorithm Stages                         */
        /*----------------------------------------------------------*/
        /* 1. Advance velocities through first half step            */
        /*----------------------------------------------------------*/
        /* !! Implement Equation (1) Loop !!                        */

        /*----------------------------------------------------------*/
#ifdef TIMING
        t0 = omp_get_wtime();
#endif
#pragma omp parallel default(none) shared(active, vel, acc, pos, delta_t, iStart, iEnd)
        {
#pragma omp for schedule(static)
            for (int istar = iStart; istar < iEnd; istar++)
            {
                if (active[istar] == 0)
                    continue;

                double* myvel = &vel[istar * 3];
                double* myacc = &acc[istar * 3];

                myvel[0] += 0.5 * myacc[0] * delta_t;
                myvel[1] += 0.5 * myacc[1] * delta_t;
                myvel[2] += 0.5 * myacc[2] * delta_t;

            } /* istar */
            /*----------------------------------------------------------*/
            /* Advance positions through a full step                    */
            /*----------------------------------------------------------*/
            /* !! Implement Equation (2) Loop !!                        */
            /*----------------------------------------------------------*/
#pragma omp for schedule(static)
            for (int istar = iStart; istar < iEnd; istar++)
            {
                if (active[istar] == 0)
                    continue;

                double* mypos = &pos[istar * 3];
                double* myvel = &vel[istar * 3];

                mypos[0] += myvel[0] * delta_t;
                mypos[1] += myvel[1] * delta_t;
                mypos[2] += myvel[2] * delta_t;

            } /* istar */
        }
#ifdef TIMING
        t_vv += omp_get_wtime() - t0;
#endif

#ifdef TIMING
        t0 = omp_get_wtime();
#endif
        MPI_Allgatherv(&pos[3 * iStart], 3 * nLocal, MPI_DOUBLE, pos, counts3, displs3, MPI_DOUBLE, MPI_COMM_WORLD);
#ifdef TIMING
        t_comm_pos += omp_get_wtime() - t0;
#endif
        /*----------------------------------------------------------*/
        /* Acceleration Calculations                                */
        /*----------------------------------------------------------*/
        /* Compute acceleration due to black hole at (0, 0, 0)      */
        /* !! Implement first term in Equation (4) !!               */
        /*----------------------------------------------------------*/
#ifdef TIMING
        t0 = omp_get_wtime();
#endif
#pragma omp parallel for default(none) shared(pos, acc, mass, active, nStars, BlackHoleMass, delta_t, iStart, iEnd)    \
    schedule(static)

        for (int istar = iStart; istar < iEnd; istar++)
        {
            if (active[istar] == 0)
                continue;

            double* mypos = &pos[istar * 3];
            double* myacc = &acc[istar * 3];

            myacc[0] = 0.0;
            myacc[1] = 0.0;
            myacc[2] = 0.0;

            double radsq = mypos[0] * mypos[0] + mypos[1] * mypos[1] + mypos[2] * mypos[2];
            double rad = sqrt(radsq);

            if (rad > 0.0)
            {
                double invr3 = 1.0 / (rad * rad * rad);
                double fac = -BigG * BlackHoleMass * invr3;

                myacc[0] += fac * mypos[0];
                myacc[1] += fac * mypos[1];
                myacc[2] += fac * mypos[2];
            }

            /*----------------------------------------------------------*/
            /* Compute accelerations due to interactions between stars. */
            /* !! Implement second term in Equation (4) !!              */
            /*----------------------------------------------------------*/
            for (int jstar = 0; jstar < nStars; jstar++)
            {
                if (jstar == istar)
                    continue;

                double* pos_j = &pos[jstar * 3];

                double dx = mypos[0] - pos_j[0];
                double dy = mypos[1] - pos_j[1];
                double dz = mypos[2] - pos_j[2];

                double radsq_ij = dx * dx + dy * dy + dz * dz;
                double rad_ij = sqrt(radsq_ij);

                if (rad_ij > 0.0)
                {
                    double invr3_ij = 1.0 / (rad_ij * rad_ij * rad_ij);
                    double fac_ij = -BigG * mass[jstar] * invr3_ij;

                    myacc[0] += fac_ij * dx;
                    myacc[1] += fac_ij * dy;
                    myacc[2] += fac_ij * dz;
                }

            } /* jstar */
        } /* istar */
#ifdef TIMING
        t_acc += omp_get_wtime() - t0;
#endif
        /*----------------------------------------------------------*/
        /* Compute drag due to dust (done for you)                  */
        /*----------------------------------------------------------*/
#ifdef TIMING
        t0 = omp_get_wtime();
#endif
#pragma omp parallel for default(none) shared(pos, vel, acc, radius, active, BHRadius, iStart, iEnd) schedule(static)
        for (int istar = iStart; istar < iEnd; istar++)
        {
            if (active[istar] == 0)
                continue;
            double* mypos = &pos[istar * 3];
            double* myvel = &vel[istar * 3];
            double* myacc = &acc[istar * 3];

            /* Compute height above black hole event horizon */
            double radsq = mypos[x] * mypos[x] + mypos[y] * mypos[y] + mypos[z] * mypos[z];
            double distance = sqrt(radsq);

            /* Compute speed */
            double velsq = myvel[x] * myvel[x] + myvel[y] * myvel[y] + myvel[z] * myvel[z];
            double speed = sqrt(velsq);

            if (speed > 0.0)
            {
                /* Get drag force - function in data.c */
                double fDrag = dragForce(radius[istar], speed, distance, BHRadius);

                /* Unit vector in direction of velocity */
                double vhat[3];
                vhat[x] = myvel[x] / speed;
                vhat[y] = myvel[y] / speed;
                vhat[z] = myvel[z] / speed;

                /* Add in acceleration due to drag */
                myacc[x] += -fDrag * vhat[x];
                myacc[y] += -fDrag * vhat[y];
                myacc[z] += -fDrag * vhat[z];
            }
        } /* istar */
#ifdef TIMING
        t_drag += omp_get_wtime() - t0;
#endif
        /*----------------------------------------------------------*/
        /* Advance velocities though second half step               */
        /*----------------------------------------------------------*/
        /* Implement Equation (3)                                   */
        /*----------------------------------------------------------*/
#ifdef TIMING
        t0 = omp_get_wtime();
#endif
#pragma omp parallel for default(none) shared(active, vel, acc, delta_t, iStart, iEnd) schedule(static)
        for (int istar = iStart; istar < iEnd; istar++)
        {
            if (active[istar] == 0)
                continue;

            double* myvel = &vel[istar * 3];
            double* myacc = &acc[istar * 3];

            myvel[0] += 0.5 * myacc[0] * delta_t;
            myvel[1] += 0.5 * myacc[1] * delta_t;
            myvel[2] += 0.5 * myacc[2] * delta_t;
        }
        /* istar */
#ifdef TIMING
        t_vv2 += omp_get_wtime() - t0;
#endif
        //}
        /* End of Velocity Verlet Step                              */
        /*==========================================================*/

#ifdef TIMING
        t0 = omp_get_wtime();
#endif
        /*----------------------------------------------/
        / Bookkeeping After VV Algorithm Step Completed /
        /----------------------------------------------*/
        double localMinradsq = DBL_MAX;
        int localActive = 0;

#ifdef TIMING
        t0 = omp_get_wtime();
#endif
#pragma omp parallel for default(none)                                                                                 \
    shared(active, pos, radius, mass, BHRadius, delta_t, istep, outfile, rank, iStart, iEnd, eventTypeLocal)           \
    reduction(min : localMinradsq) reduction(+ : localActive) schedule(static)
        for (int istar = iStart; istar < iEnd; istar++)
        {
            /* Don't check stars which have crashed */

            if (active[istar] == 0)
                continue;

            double* mypos = &pos[istar * 3];

            /* Compute radius and store if the smallest this step */
            double radsq = mypos[x] * mypos[x] + mypos[y] * mypos[y] + mypos[z] * mypos[z];
            if (radsq < localMinradsq)
                localMinradsq = radsq;

            /* Check for encountering of event horizon or escape */
            double r = sqrt(radsq);
            int check_minrad = r + radius[istar] < BHRadius;
            int check_maxrad = r > EscapeRadius;
            if (check_minrad || check_maxrad)
            {
                active[istar] = 0; /* Deactivate star */
                eventTypeLocal[istar] = check_minrad ? 0 : 1;
            }
            else
            {
                localActive++;
            }
        }
#ifdef TIMING
        t_book += omp_get_wtime() - t0;
#endif

#ifdef TIMING
        t0 = omp_get_wtime();
#endif
        MPI_Reduce(eventTypeLocal, eventTypeRoot, nStars, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Allgatherv(&active[iStart], nLocal, MPI_INT, active, counts1, displs1, MPI_INT, MPI_COMM_WORLD);

        MPI_Allreduce(&localActive, &nActive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&localMinradsq, &minRadsq, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

#ifdef TIMING
        t_comm_misc += omp_get_wtime() - t0;
#endif
        // if (rank == 0 && istep % s == 0)
        // {
        //     for (int istar = 0; istar < nStars; istar++)
        //     {
        //         if (active[istar] == 0)
        //             continue;
        //         printf("ORBIT %d %d %e %e %e\n", istep, istar, pos[istar * 3 + 0], pos[istar * 3 + 1],
        //                pos[istar * 3 + 2]);
        //     }
        // }

        if (rank == 0)
        {
            /* Report on Escaping or Swallowed Stars */
            int elapsedTime = (int)(delta_t * (double)istep / SecondsIn1Myear);
            for (int istar = 0; istar < nStars; istar++)
            {
                if (eventTypeRoot[istar] == 0)
                {
                    printf("# At %d Star %d of mass %12.6e MSun encountered the Black Hole event horizon\n",
                           elapsedTime, istar, mass[istar] / MSun);
                    printf("# -------------------------------\n");

                    fprintf(outfile, "Step %d : Star of mass %12.6f Swallowed\n", istep, mass[istar] / MSun);

                    fflush(outfile);
                }
                else if (eventTypeRoot[istar] == 1)
                {
                    printf("# Star %d has escaped the simulation!\n", istar);
                    printf("# -------------------------------\n");

                    fprintf(outfile, "Step %d : Star of mass %12.6f Escaped\n", istep, mass[istar] / MSun);

                    fflush(outfile);
                }
            }
        }

/*-----------------------------------------------------------------*/
/* Output some info to screen every 100 steps                      */
/* - Appropriate place to add any other intermediate output wanted */
/*-----------------------------------------------------------------*/
#ifdef TIMING
        t0 = omp_get_wtime();
#endif
        if (rank == 0 && istep % 100 == 0)
        {
            printf("Steps elapsed : %8d, remaining stars : %8d, nearest object at distance of %20.3f Ly (%20.3f RBH)\n",
                   istep, nActive, sqrt(minRadsq) / LightYear, sqrt(minRadsq) / BHRadius);
        }

        if (nActive < 1)
        {
            if (rank == 0)
                printf("# No objects left to track!\n");
            break;
        }
#ifdef TIMING
        t_out += omp_get_wtime() - t0;
#endif

    } /* istep */
#ifdef TIMING
    double sum_vv, sum_comm_pos, sum_acc, sum_book, sum_out, sum_comm_misc, sum_drag, sum_vv2;

    MPI_Reduce(&t_vv, &sum_vv, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_pos, &sum_comm_pos, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_acc, &sum_acc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_book, &sum_book, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_misc, &sum_comm_misc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_out, &sum_out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_drag, &sum_drag, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_vv2, &sum_vv2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        FILE* tf = fopen("timing.dat", "w");
        if (tf)
        {
            fprintf(tf, "# TIMING SUMMARY (MAX over ranks)\n");
            fprintf(tf, "VV_total      %e\n", sum_vv);
            fprintf(tf, "ACC_total     %e\n", sum_acc);
            fprintf(tf, "DRAG_total    %e\n", sum_drag);
            fprintf(tf, "VV2_total     %e\n", sum_vv2);
            fprintf(tf, "BOOK_total    %e\n", sum_book);
            fprintf(tf, "OUTPUT_total  %e\n", sum_out);
            fprintf(tf, "COMM_POS_total %e\n", sum_comm_pos);
            fprintf(tf, "COMM_MISC_total %e\n", sum_comm_misc);
            fclose(tf);
        }
    }
#endif
    /*--------------------------*/
    /*   Stop Overall timer     */
    /*--------------------------*/
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = omp_get_wtime();
    double localTime = t2 - t1;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    //     printf("# Total Simulation Time : %f seconds\n", maxTime);
    /*-----------*/
    /* Clean up  */
    /*-----------*/
    free(active);
    free(mass);
    free(radius);
    free(pos);
    free(vel);
    free(acc);
    free(counts3);
    free(displs3);
    free(counts1);
    free(displs1);
    free(eventTypeLocal);
    if (rank == 0)
    {
        free(eventTypeRoot);
        if (outfile)
            fclose(outfile);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}