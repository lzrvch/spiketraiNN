#ifndef DIST_H_
#define DIST_H_

void setDistanceParameters(double parameter_a, double parameter_b, int nSpikes1, int nSpikes2);

void initializeArrays(double* SpikeTrain1, double* SpikeTrain2, double* distance);

void computeVPDistance();
void computeSrbrDistance();
void computeISIDistance();
void computeSPIKEDistance();
void computeVRDistance();
void computeMaxMetricDistance();
void computeDTWDistance();
void computeModulusMetricDistance();

#endif /* DIST_H_ */
