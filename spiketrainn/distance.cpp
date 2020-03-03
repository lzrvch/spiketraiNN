#include <bits/stdc++.h> 
#include <vector>
#include <tuple>
#include <iostream>
#include <algorithm>
#include "distance.h"
#include "dtw.h"

typedef std::vector<double> SPIKE_TRAIN;


namespace spikeDist {
    double* SpikeTrain_1;
    double* SpikeTrain_2;
    double* distance_value;
    double parameter_a;
    double parameter_b;
    int nSpikes_1;
    int nSpikes_2;
}

using namespace spikeDist;

void setDistanceParameters(double parameter_a, double parameter_b, int nSpikes1, int nSpikes2){
    spikeDist::parameter_a = parameter_a;
    spikeDist::parameter_b = parameter_b;
    spikeDist::nSpikes_1 = nSpikes1;
    spikeDist::nSpikes_2 = nSpikes2;
}

void initializeArrays(double* SpikeTrain1, double* SpikeTrain2, double* distance){
    spikeDist::SpikeTrain_1 = SpikeTrain1;
    spikeDist::SpikeTrain_2 = SpikeTrain2;
    spikeDist::distance_value = distance;
}

double vp_measure(SPIKE_TRAIN T1, SPIKE_TRAIN T2, double q) {
    // T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    size_t n1 = T1.size();
    size_t n2 = T2.size();

    ++n1; ++n2;

    std::vector<std::vector<double> > DD(n1, std::vector<double>(n2,0));

    double zet = 0.;

    for (unsigned int i = 0; i < n1; ++i)
       DD[i][0] = i;

    for (unsigned int i = 0; i < n2; ++i)
       DD[0][i] = i;

    for (unsigned int i = 1; i < n1; ++i)
       for (unsigned int j = 1; j < n2; ++j) {
           zet = q * fabs(T1[i-1]-T2[j-1]);
           if (((DD[i-1][j]) <= (DD[i][j-1])) && ((DD[i-1][j] + 1) <= (DD[i-1][j-1] + zet)))
             DD[i][j]=DD[i-1][j] + 1.;
           else if ((DD[i][j-1] + 1) <= (DD[i-1][j-1] + zet))
                  DD[i][j]=DD[i][j-1] + 1;
                else
                  DD[i][j] = DD[i-1][j-1] + zet;
       }

    return DD[n1-1][n2-1];
}

void computeVPDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    distance_value[0] = vp_measure(T1, T2, parameter_a);   
}

double srbr_measure (SPIKE_TRAIN T1, SPIKE_TRAIN T2, double a, double b) {
    //T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    double dt = 1; //the integration timestep
    double ds_T12 = 0, ds_T1 = 0, ds_T2 = 0;

    unsigned int j1 = 0, j2 = 0;
    double E1 = 0, E2 = 0;

    double sig = 10; //the width of the Gaussian kernel
    double sig2 = 2*sig*sig;

    unsigned int n1 = T1.size ();
    unsigned int n2 = T2.size ();

    for (double i = a; i < b; i+=dt) {
        if (j1 < n1 && T1[j1] <= i) ++j1;
        if (j2 < n2 && T2[j2] <= i) ++j2;

        E1 = 0; E2 = 0;
        for (unsigned int j = 0; j < j1; ++j)
            E1 += std::exp ( - pow ((i  - T1[j]), 2) / sig2);

        for (unsigned int j = 0; j < j2; ++j)
            E2 += std::exp (- pow ((i  - T2[j]), 2) / sig2);

        ds_T12 += E1*E2;
        ds_T1 += E1*E1;
        ds_T2 += E2*E2;
    }
    return 1 - ds_T12 / (sqrt(ds_T1 * ds_T2));
}

void computeSrbrDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    distance_value[0] = srbr_measure(T1, T2, parameter_a, parameter_b);   
}

double k_isi_measure (SPIKE_TRAIN T1, SPIKE_TRAIN T2, double a, double b) {
    //T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    //T1 does not contain overlapping spikes, nor does T2
  //a < T1,2[i] and b > T1,2[i]
  double dt = 1; //the integration timestep
    double dI = 0, I = 0;

    double x_ISI_1 = 0, x_ISI_2 = 0;

    //add to each spike train an auxiliary leading spike at time t=a (the beginning of the recording)
    //and an auxiliary trailing spike at time t=b (the end of the recording)
    T1.insert (T1.begin (), a); T1.insert (T1.end (), b);
    T2.insert (T2.begin (), a); T2.insert (T2.end (), b);

    //i1 is the index of the first spike in T1 U {a,b} having a timing that is greater than t
    //i2 is computed analogously for T2 U {a,b}
    int i1 = 0, i2 = 0;

    unsigned int n1 = T1.size ();
    unsigned int n2 = T2.size ();

    for (double t = a; t < b; t+=dt){

        while (t >= T1[i1] && i1 < n1 - 1)
            i1 += 1;
        while (t >= T2[i2] && i2 < n2 - 1)
            i2 += 1;

    //now T1[i1] > t and T1[i1-1] <= t
    //the same for i2 and T2
        x_ISI_1 = T1[i1] - T1[i1-1]; x_ISI_2 = T2[i2] - T2[i2-1];

        if (x_ISI_1 <= x_ISI_2)
            I = x_ISI_1 / x_ISI_2 - 1;
        else
            I = - (x_ISI_2 / x_ISI_1 - 1);

        dI += fabs (I)*dt;
    }

    return dI / (b - a);
}

void computeISIDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    distance_value[0] = k_isi_measure(T1, T2, parameter_a, parameter_b);   
}

double k_spk_measure (SPIKE_TRAIN T1, SPIKE_TRAIN T2, double a, double b) {
    //T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    //T1 does not contain overlapping spikes, nor does T2
  //a < T1,2[i] and b > T1,2[i].
    double dt = 1; //the integration timestep
    double dS = 0, S = 0, S_1 = 0, S_2 = 0;

    double x_ISI_1, x_ISI_2;

    double delta_t_P_1, delta_t_F_1;
    double delta_t_P_2, delta_t_F_2;

    double x_P_1,  x_F_1;
    double x_P_2,  x_F_2;

    //add to each spike train an auxiliary leading spike at time t=a (the beginning of the recording)
    //and an auxiliary trailing spike at time t=b (the end of the recording)
    T1.insert (T1.begin (), a); T1.insert (T1.end (), b);
    T2.insert (T2.begin (), a); T2.insert (T2.end (), b);

    //i1 is the index of the first spike in T1 U {a,b} having a timing that is greater than t
    //i2 is computed analogously for T2 U {a,b}
    int i1 = 0, i2 = 0;

    unsigned int n1 = T1.size ();
    unsigned int n2 = T2.size ();

    for (double t = a; t < b; t+=dt){

        while (t >= T1[i1] && i1 < n1 - 1)
            i1 += 1;
        while (t >= T2[i2] && i2 < n2 - 1)
            i2 += 1;
    //now T1[i1] > t and T1[i1-1] <= t
    //the same for i2 and T2

        x_ISI_1 = T1[i1] - T1[i1-1]; x_ISI_2 = T2[i2] - T2[i2-1];

        delta_t_P_1 = b - a;
        delta_t_F_1 = b - a;
        delta_t_F_2 = b - a;
        delta_t_P_2 = b - a;


        for (unsigned int i = i2; i >= 0 && fabs(T1[i1-1] - T2[i])  < delta_t_P_1; delta_t_P_1 = fabs(T1[i1-1] - T2[i]), --i);
        for (unsigned int i = i2 - 1; i < n2 && fabs(T1[i1] - T2[i])  < delta_t_F_1; delta_t_F_1 = fabs(T1[i1] - T2[i]), ++i);
        for (unsigned int i = i1; i >= 0 && fabs(T2[i2-1] - T1[i])  < delta_t_P_2; delta_t_P_2 = fabs(T2[i2-1] - T1[i]), --i);
        for (unsigned int i = i1 - 1; i < n1 && fabs(T2[i2] - T1[i])  < delta_t_F_2; delta_t_F_2 = fabs(T2[i2] - T1[i]), ++i);

        x_P_1 = t - T1[i1-1]; x_P_2 = t - T2[i2-1];
        x_F_1 = T1[i1] - t; x_F_2 = T2[i2] - t;

        S_1 = (delta_t_P_1 * x_F_1 + delta_t_F_1 * x_P_1) / x_ISI_1;
        S_2 = (delta_t_P_2 * x_F_2 + delta_t_F_2 * x_P_2) / x_ISI_2;

        S = (S_1 * x_ISI_2 + S_2 * x_ISI_1) / (2 * pow((x_ISI_1+ x_ISI_2) / 2.0, 2));

        dS += S*dt;
    }

    return dS / (b - a);
}

void computeSPIKEDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    distance_value[0] = k_spk_measure(T1, T2, parameter_a, parameter_b);   
}

double vr_measureOP (SPIKE_TRAIN T1, SPIKE_TRAIN T2, double a, double b) {
    //T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    //T1 does not contain overlapping spikes, nor does T2
    double dt = 1; //the integration timestep
    double dv = 0;

    unsigned int j1 = 0, j2 = 0;
    double f = 0, g = 0;
    double tau = 10;

    int i1 = 0, i2 = 0;

    int n1 = T1.size ();
    int n2 = T2.size ();

  double e;


    for (double t = a; t < b; t+=dt) {

        e = std::exp (-dt/tau);
        f = f * e;
        g = g * e;

        while ((i1 < n1) && (T1[i1] > (t) && T1[i1] <= (t + dt))) {
            f += std::exp (-(t+dt-T1[i1])/tau);
            i1 += 1;
        }

        while ((i2 < n2) && (T2[i2] > (t) && T2[i2] <= (t + dt))) {
            g += std::exp (-(t+dt-T2[i2])/tau);
            i2 += 1;
        }

        dv += (f-g)*(f-g)*dt;
    }

    return dv;
}

void computeVRDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    distance_value[0] = vr_measureOP(T1, T2, parameter_a, parameter_b);   
}


double dm_measure (SPIKE_TRAIN T1, SPIKE_TRAIN T2, double a, double b) {
    //T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    //T1 does not contain overlapping spikes, nor does T2
  float dt = 1; //the integration timestep
  double dm = 0;

  double E1 = 0, E2 = 0;
  double M1 = 0, M2 = 0;

  double Max = 0;
  double MS = 0;

    unsigned int n1 = T1.size ();
    unsigned int n2 = T2.size ();

    //i1 is the index of the first spike in T1 U {a,b} having a timing that is greater than t
    //i2 is computed analogously for T2 U {a,b}
    int i1 = 0, i2 = 0;

    //add to each spike train an auxiliary leading spike at time t=a (the beginning of the recording)
    //and an auxiliary trailing spike at time t=b (the end of the recording)
    //T1.insert (T1.begin (), a); T1.insert (T1.end (), b);
    //T2.insert (T2.begin (), a); T2.insert (T2.end (), b);

  for (double i = a; i < b; i+=dt) {
    Max = 0;
        i1 = 0; i2 = 0;

    for (double t = a; t < b; t+=dt) {
            while (t >= T1[i1] && i1 < n1 - 1)
                i1 += 1;
            while (t >= T2[i2] && i2 < n2 - 1)
                i2 += 1;

            if (i1 == 0) //left corner
                M1 = T1[i1] - t;
            else if (i1 == n1 - 1) //right corner
                M1 = t - T1[i1-1];
            else
                M1 = std::min (T1[i1] - t,t - T1[i1-1]);

            if (i2 == 0) //left corner
                M2 = T2[i2] - t;
            else if (i2 == n2 - 1) //right corner
                M2 = t - T2[i2-1];
            else
                M2 =  std::min (T2[i2] - t,t - T2[i2-1]);

            MS = fabs(M1 - M2) * std::exp(-abs (i-t)/10.0);

            if (MS > Max) Max = MS;
    }

    dm += Max*dt;
  }

  return dm;
}

void computeMaxMetricDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    distance_value[0] = dm_measure(T1, T2, parameter_a, parameter_b);   
}

void computeDTWDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    LB_Improved filter(T1, nSpikes_1 / 10); 
    distance_value[0] = filter.test(T2);   
}


double d(int t, SPIKE_TRAIN &T, int i){
        //Input:  a timing t, a sorted spike train T and an index i of a spike in T
        //        such that either t <= T[i] or i is the index of the last spike of T
        //Output: the distance d(t, T) between a timing t and a spike train T
        double db = abs(T[i] - t);
        int j = i - 1;
        while (j >= 0 && abs(T[j] - t) <= db) {
            db = abs(T[j] - t);
            j -= 1;
        }
        return db;
}

double do_measure_optimal (SPIKE_TRAIN T1, SPIKE_TRAIN T2, double a, double b) {
    using namespace std;

    //T1 and T2 are ordered, nonempty sets of real numbers, indexed starting from 0.
    //i1 and i2 are the indices of the currently processed spikes in the two spike
    //trains. p1 and p2 are the indices of the previously processed spikes in the
    //two spike trains. p is the index of the spike train to which the previously
    //processed spike belonged (1 or 2), after at least one spike has been processed,
    //or 0 otherwise.

    int i1 = 0, i2 = 0, p1 = 0, p2 = 0, p = 0;
    int n1 = T1.size (), n2 = T2.size ();

    //P is an array of structures (s,phi) consisting of a ordered pair of numbers.
    vector <tuple<double,double> > P;
    P.reserve(3*(n1 + n2));
    P.push_back (make_tuple (a, abs(T1[0] - T2[0])));

    double t = 0.0;
    //Process the spikes until the end of one of the spikes trains is reached.
    while (i1 < n1 && i2 < n2) {
        if (T1[i1] <= T2[i2]){
            if (i1 > 0) {
                //Adds to P the timing situated at the middle of the interval between the
                //currently processed spike and the previous spike in the same spike train.
                t = (T1[i1] + T1[i1 - 1]) / 2.0;
                //We have d(t, T1) = T1[i1] - t = t - T1[i1 - 1] = (T1[i1] - T1[i1 - 1]) / 2.
                P.push_back (make_tuple (t, abs((T1[i1] - T1[i1 - 1]) / 2.0 - d(t,T2,i2))));
            }
            if (p == 2) {
                //If the previously processed spike was one from the other spike train than
                //the spike currently processed, adds to P the timing situated at the
                //middle of the interval between the currently processed spike and the
                //previously processed spike.
                t = (T1[i1] + T2[p2]) / 2.0;
                //Since t is is at equal distance to the closest spikes in the two spike
                //trains, T1[i1] and T2[p2], we have d(t, T1)=d(t, T2) and phi(t)=0.
                P.push_back (make_tuple (t, 0));
            }
            //Adds to P the currently processed spike.
            t = T1[i1];
            //We have d(t, T1) = 0. If at least one spike from T2 has been processed, we
            //have T2[p2] <= t <= T2[i2], with i2 = p2 + 1, and thus
            //d(t, T2) = min(|t-T2[p2]|, T2[i2]-t). If no spike from T2 has been processed,
            //we have p2 = i2 = 0, and the previous formula for d(t, T2) still holds.
            P.push_back (make_tuple (t, min(abs(t - T2[p2]), T2[i2] - t)));
            p1 = i1;
            i1 += 1;
            p = 1;
        }
        else {
            //Proceed analogously for the case T1[i1] > T2[i2]:
            if (i2 > 0) {
                t = (T2[i2] + T2[i2 - 1]) / 2.0;
                P.push_back (make_tuple (t, abs((T2[i2] - T2[i2 - 1]) / 2.0 - d(t,T1,i1))));
            }
            if (p == 1) {
                t = (T2[i2] + T1[p1]) / 2.0;
                P.push_back (make_tuple (t, 0));
            }
            t = T2[i2];
            P.push_back (make_tuple (t, min(abs(t - T1[p1]), T1[i1] - t)));
            p2 = i2;
            i2 += 1;
            p = 2;
        }
    }
    //Process the rest of the spikes in the spike train that has not been fully
    //processed:
    while (i1 < n1) {
            if (i1 > 0) {
                //Adds to P the timing situated at the middle of the interval between the
                //currently processed spike and the previous spike in the same spike train
                t = (T1[i1] + T1[i1 - 1]) / 2.0;
                //We have d(t, T1) = T1[i1] - t = t - T1[i1 - 1] = (T1[i1] - T1[i1 - 1]) / 2.
                P.push_back (std::make_tuple (t, abs((T1[i1] - T1[i1 - 1]) / 2.0 - d(t,T2,p2))));
            }
            if (p == 2) {
                //If the previously processed spike was one from the other spike train than
                //the spike currently processed (i.e., the last spike in the spike train
                //that has been fully processed), adds to P the timing situated at the
                //middle of the interval between the currently processed spike and the
                //previously processed spike.
                t = (T1[i1] + T2[p2]) / 2.0;
                //Since t is is at equal distance to the closest spikes in the two spike
                //trains, T1[i1] and T2[p2], we have d(t, T1)=d(t, T2) and phi(t)=0.
                P.push_back (std::make_tuple (t, 0));
            }
            //Adds to P the currently processed spike.
            t = T1[i1];
            //We have d(t, T1) = 0. We have T2[p2] <= t and the spike at p2 ist the last one
            //in T2, and thus d(t, T2) = t - T2[p2].
            P.push_back (std::make_tuple (t, t - T2[p2]));
            //p1 = i1 #This could be added for completeness, but it is not used by this algorithm.
            i1 += 1;
            p = 1;
    }
    while  (i2 < n2){
            //Proceed analogously for the case that the train that has not been fully processed
            //is T2:
            if (i2 > 0) {
                t = (T2[i2] + T2[i2 - 1]) / 2.0;
                P.push_back (std::make_tuple (t, abs((T2[i2] - T2[i2 - 1]) / 2.0 - d(t,T1,p1))));
            }
            if (p == 1) {
                t = (T2[i2] + T1[p1]) / 2.0;
                P.push_back (std::make_tuple (t, 0));
            }
            t = T2[i2];
            P.push_back (std::make_tuple (t, t - T1[p1]));
            //p2 = i2
            i2 += 1;
            p = 2;
    }
    P.push_back (std::make_tuple (b, abs(T1[n1 - 1] - T2[n2 - 1])));
    //sort P with regard to the first element
    sort (P.begin (), P.end());

    unsigned int psize = P.size ();

    //perform the integration
    double dO = 0.0;
    for (unsigned int i = 1; i < psize; ++i)
        dO += (std::get<0>(P[i]) - std::get<0>(P[i-1])) * (std::get<1>(P[i]) + std::get<1>(P[i-1])) / 2.0;

    return dO;
}

void computeModulusMetricDistance(){
    std::vector<double> T1(SpikeTrain_1, SpikeTrain_1 + nSpikes_1);
    std::vector<double> T2(SpikeTrain_2, SpikeTrain_2 + nSpikes_2);
    distance_value[0] = do_measure_optimal(T1, T2, parameter_a, parameter_b);   
}
