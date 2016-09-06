#include <math.h>
#include <stdlib.h>
#include "mpi.h"

#include "integral.h"

const double EPSILON = 0.0001;
const double CIRCLE_RADIUS = 1;
const int    MASTER_RANK = 0;

int main(int argc, char* argv[])
{

	int numberOfWorkers, currentId;
	double startTime, stopTime, result;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentId);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfWorkers);

	if (numberOfWorkers < 2)
	{
		printf("Program requires at least 2 nodes");
		MPI_Finalize();
		exit(1);
	}


	if (currentId == MASTER_RANK) // Master Work
	{
		startTime = MPI_Wtime();

		result = masterWork(numberOfWorkers);

		stopTime = MPI_Wtime();

		printf("PI: %f\n", 4 * result);
		printf("Execution Time: %f\n", stopTime - startTime);
	}
	else //Slave work
	{
		slaveWork();
	}

	MPI_Finalize();
	return 0;
}

double masterWork(int numberOfWorkers)
{
	Task task = { 0, 0, CIRCLE_RADIUS, EPSILON };
	MPI_Status status;
	double totalSum = 0;

	createAndCommitTaskDataType(&task);

	Range* ranges = prepareRanges(numberOfWorkers-1);
	// Send Tasks to slaves
	for (int i = 1; i < numberOfWorkers; i++)
	{
		task.from = ranges[i-1].from;
		task.to   = ranges[i-1].to;
		MPI_Send(&task, 1, MPI_TaskDataType, i, 0, MPI_COMM_WORLD);
	}
	delete[] ranges;

	for (int i = 1; i < numberOfWorkers; i++)
	{
		double receivedSum;
		MPI_Recv(&receivedSum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		totalSum += receivedSum;
	}

	return totalSum;
}

void slaveWork()
{
	Task task;
	MPI_Status status;
	double totalSum = 0;

	createAndCommitTaskDataType(&task);

	MPI_Recv(&task, 1, MPI_TaskDataType, MASTER_RANK, 0, MPI_COMM_WORLD, &status);


	totalSum = calculateIntegral(&task);

	MPI_Send(&totalSum, 1, MPI_DOUBLE, MASTER_RANK, 0, MPI_COMM_WORLD);
}

Range* prepareRanges(int numberOfWorkers)
{
	double segmentSize = CIRCLE_RADIUS / (double)numberOfWorkers;
	Range* ranges = new Range[numberOfWorkers];

	for (int i = 0; i < numberOfWorkers; i++)
	{
		ranges[i].from = segmentSize*i;
		ranges[i].to = ranges[i].from + segmentSize;
	}
	return ranges;
}

double calculateIntegral(Task* task)
{
	double result  = 0;
	double from    = task->from;
	double to      = task->to;
	double radius  = task->circleRadius;
    double epsilon = task->epsilon;

	for (double i = from; i < to; i += epsilon)
	{
		result += epsilon * sqrt(pow(radius, 2) - pow(i, 2));
	}
	return result;
}

void createAndCommitTaskDataType(Task* task)
{
	MPI_Datatype innerTypes[4] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	MPI_Aint addressesOfInnerTypes[4];
	int innerTypeLen[4] = { 1, 1, 1, 1 };

	// Calculate address offsets
	addressesOfInnerTypes[0] = (char*)&task->from         - (char*)task;
	addressesOfInnerTypes[1] = (char*)&task->to           - (char*)task;
	addressesOfInnerTypes[2] = (char*)&task->circleRadius - (char*)task;
    addressesOfInnerTypes[3] = (char*)&task->epsilon      - (char*)task;


    // Create and commit data type binding
	MPI_Type_create_struct(4, innerTypeLen, addressesOfInnerTypes, innerTypes, &MPI_TaskDataType);
	MPI_Type_commit(&MPI_TaskDataType);

}