#ifndef __INTEGRAL_H
#define __INTEGRAL_H
#include "mpi.h"

MPI_Datatype MPI_TaskDataType;
enum TASK_TYPE { STOP_EXECUTION, EXECUTE };

struct Range {
	double from;
	double to;
};

struct Task {
	double    from;
	double    to;
	double    circleRadius;
	double    epsilon;
};

void createAndCommitTaskDataType(Task* task);
void slaveWork();
double masterWork(int numberOfWorkers);
Range* prepareRanges(int numberOfWorkers);
double calculateIntegral(Task* task);

#endif