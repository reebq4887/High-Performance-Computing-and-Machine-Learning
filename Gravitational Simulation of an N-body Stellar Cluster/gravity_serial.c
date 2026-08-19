/*========================================================/
/   PX457 Assignment 5 2025 - Galactic Dynamics           /
/                                                         /
/        Skeleton code with some missing elements         /
/                                                         /
/               B. Morgan - December 2025                 /
/               Original code from N.D.M Hine             /
/========================================================*/
#include "data.h"
#include "mt19937ar.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

    int istep, istar, jstar; /* Loop counters */
    double minRadsq;         /* Squared radius of closest star */

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
        printf("Error Processing command line arguments\n");
        return EXIT_FAILURE;
    }
    printf("# Command line arguments successfully processed\n");

    printf("# BHRadius: %e\n", BHRadius / LightYear);
    printf("# delta_t: %e\n", delta_t / SecondsIn1Myear);
    printf("# ClusterMass: %e\n", ClusterMass / MSun);
    printf("# ClusterRadius: %e\n", ClusterRadius / LightYear);
    printf("# ClusterApogee: %e\n", ClusterApogee / LightYear);
    printf("# ClusterPerigee: %e\n", ClusterPerigee / LightYear);
    printf("# KE_init: %e\n", KE_init / KE_units);

    /*------------------/
    / Allocate memory   /
    /------------------*/
    /* Allocate memory for mass and radius */
    mass = (double*)malloc(nStars * sizeof(double));
    radius = (double*)malloc(nStars * sizeof(double));
    if (mass == NULL)
    {
        printf("Error allocating mass");
        exit(EXIT_FAILURE);
    }
    if (radius == NULL)
    {
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
        printf("Error allocating pos");
        exit(EXIT_FAILURE);
    }
    if (vel == NULL)
    {
        printf("Error allocating vel");
        exit(EXIT_FAILURE);
    }
    if (acc == NULL)
    {
        printf("Error allocating acc");
        exit(EXIT_FAILURE);
    }

    /* Array indicating if a star is active in the simulation */
    active = (int*)malloc(nStars * sizeof(int));
    if (active == NULL)
    {
        printf("Error allocating active");
        exit(EXIT_FAILURE);
    }

    /*------------------------*/
    /* Initialise the stars   */
    /*------------------------*/
    initialiseCluster(nStars, KE_init, ClusterMass, BlackHoleMass, ClusterRadius, ClusterApogee, ClusterPerigee, mass,
                      radius, pos, vel, acc);

    /* Active simulation elements */
    int nActive = nStars;
    for (istar = 0; istar < nStars; istar++)
    {
        active[istar] = 1;
    }

    /*------------------------*/
    /* Print some diagnostics */
    /*------------------------*/
    printf("# Generated %d stars\n", nStars);
    for (istar = 0; istar < nStars; istar++)
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

    /*------------------*/
    /* Open output file */
    /*------------------*/
    outfile = fopen("output.dat", "w");
    if (outfile == NULL)
    {
        printf("Error opening output file!\n");
        exit(EXIT_FAILURE);
    }

    /*--------------------------------*/
    /* !!Start Overall timer here!!   */
    /*--------------------------------*/

    /*---------------------------*/
    /* Outer loop over timesteps */
    /*---------------------------*/

    const int s = 100; /* Output interval */

    for (istep = 0; istep < Nsteps; istep++)
    {
        /* Initialise distance to closest star */
        minRadsq = DBL_MAX;

        /*==========================================================*/
        /* Velocity Verlet Algorithm Stages                         */
        /*----------------------------------------------------------*/
        /* 1. Advance velocities through first half step            */
        /*----------------------------------------------------------*/
        /* !! Implement Equation (1) Loop !!                        */
        /*----------------------------------------------------------*/
        for (istar = 0; istar < nStars; istar++)
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
        for (istar = 0; istar < nStars; istar++)
        {
            if (active[istar] == 0)
                continue;

            double* mypos = &pos[istar * 3];
            double* myvel = &vel[istar * 3];

            mypos[0] += myvel[0] * delta_t;
            mypos[1] += myvel[1] * delta_t;
            mypos[2] += myvel[2] * delta_t;

        } /* istar */

        /*----------------------------------------------------------*/
        /* Acceleration Calculations                                */
        /*----------------------------------------------------------*/
        /* Compute acceleration due to black hole at (0, 0, 0)      */
        /* !! Implement first term in Equation (4) !!               */
        /*----------------------------------------------------------*/
        for (istar = 0; istar < nStars; istar++)
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
            for (jstar = 0; jstar < nStars; jstar++)
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

        /*----------------------------------------------------------*/
        /* Compute drag due to dust (done for you)                  */
        /*----------------------------------------------------------*/
        for (istar = 0; istar < nStars; istar++)
        {
            double* mypos = &pos[istar * 3];
            double* myvel = &vel[istar * 3];
            double* myacc = &acc[istar * 3];

            /* Compute height above black hole event horizon */
            double radsq = mypos[x] * mypos[x] + mypos[y] * mypos[y] + mypos[z] * mypos[z];
            double distance = sqrt(radsq);

            /* Compute speed */
            double velsq = myvel[x] * myvel[x] + myvel[y] * myvel[y] + myvel[z] * myvel[z];
            double speed = sqrt(velsq);

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

        } /* istar */

        /*----------------------------------------------------------*/
        /* Advance velocities though second half step               */
        /*----------------------------------------------------------*/
        /* Implement Equation (3)                                   */
        /*----------------------------------------------------------*/
        for (istar = 0; istar < nStars; istar++)
        {
            if (active[istar] == 0)
                continue;

            double* myvel = &vel[istar * 3];
            double* myacc = &acc[istar * 3];

            myvel[0] += 0.5 * myacc[0] * delta_t;
            myvel[1] += 0.5 * myacc[1] * delta_t;
            myvel[2] += 0.5 * myacc[2] * delta_t;
        } /* istar */

        /* End of Velocity Verlet Step                              */
        /*==========================================================*/

        // if (istep % s == 0)
        // {
        //     for (int istar = 0; istar < nStars; istar++)
        //     {
        //         if (active[istar] == 0)
        //             continue;
        //         printf("ORBIT %d %d %e %e %e\n", istep, istar, pos[istar * 3 + 0], pos[istar * 3 + 1],
        //                pos[istar * 3 + 2]);
        //     }
        // }

        /*----------------------------------------------/
        / Bookkeeping After VV Algorithm Step Completed /
        /----------------------------------------------*/
        for (istar = 0; istar < nStars; istar++)
        {
            /* Don't check stars which have crashed */
            if (active[istar] == 0)
                continue;

            double* mypos = &pos[istar * 3];

            /* Compute radius and store if the smallest this step */
            double radsq = mypos[x] * mypos[x] + mypos[y] * mypos[y] + mypos[z] * mypos[z];
            if (radsq < minRadsq)
                minRadsq = radsq;

            /* Check for encountering of event horizon or escape */
            int check_minrad = sqrt(radsq) + radius[istar] < BHRadius;
            int check_maxrad = sqrt(radsq) > EscapeRadius;
            if (check_minrad || check_maxrad)
            {
                /* Report on Escaping or Swallowed Stars */
                int elapsedTime = (int)(delta_t * (double)istep / SecondsIn1Myear);
                if (check_minrad)
                {
                    printf("# At %d Star %d of mass %12.6e MSun encountered the Black Hole event horizon\n",
                           elapsedTime, istar, mass[istar] / MSun);
                }
                else if (check_maxrad)
                {
                    printf("# Star %d has escaped the simulation!\n", istar);
                }
                printf("# -------------------------------\n");

                active[istar] = 0; /* Track this star no more */
                nActive--;         /* Reduce number of stars  */
                printf("\n");

                /* Output to file with mass at encounter */
                if (check_minrad)
                {
                    fprintf(outfile, "Step %d : Star of mass %12.6f Swallowed\n", istep, mass[istar] / MSun);
                }
                if (check_maxrad)
                {
                    fprintf(outfile, "Step %d : Star of mass %12.6f Escaped\n", istep, mass[istar] / MSun);
                }
                fflush(outfile);

            } /* output */

        } /* istar */

        /*-----------------------------------------------------------------*/
        /* Output some info to screen every 100 steps                      */
        /* - Appropriate place to add any other intermediate output wanted */
        /*-----------------------------------------------------------------*/
        if (istep % 100 == 0)
        {
            printf("Steps elapsed : %8d, remaining stars : %8d, nearest object at distance of %20.3f Ly (%20.3f RBH)\n",
                   istep, nActive, sqrt(minRadsq) / LightYear, sqrt(minRadsq) / BHRadius);
        }

        if (nActive < 1)
        {
            printf("# No objects left to track!\n");
            break;
        }

    } /* istep */

    /*--------------------------*/
    /*   Stop Overall timer     */
    /*--------------------------*/

    /*-----------*/
    /* Clean up  */
    /*-----------*/
    free(active);
    free(mass);
    free(radius);
    free(pos);
    free(vel);
    free(acc);

    fclose(outfile);

    return EXIT_SUCCESS;
}