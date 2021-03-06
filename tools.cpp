#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd RMSE;
	if (estimations.size() == ground_truth.size())
	{
		int size = estimations.size();
		if (size != 0)
		{
			if (estimations[0].size() == ground_truth[0].size())
			{
				int dim = ground_truth[0].size();
				RMSE = VectorXd(dim);

				//Summarize error squares
				for (int d = 0; d<dim; d++)
				{
					RMSE(d) = 0;
				}
				for (int s = 0; s<size; s++)
				{
					if ((estimations[s].size() >= dim) && (ground_truth[s].size() >= dim))
					{
						VectorXd residual = estimations[s] - ground_truth[s];
						residual = residual.array()*residual.array();
						RMSE += residual;
					}
					else
					{
						std::cout << "ERROR: CalculateRMSE Dimensions of estimation or ground_truth not correct!\n";
					}
				}

				//Mean
				RMSE = RMSE / size;
				//Square root
				RMSE = RMSE.array().sqrt();
			}
		}
	}
	else
	{
		std::cout << "ERROR: CalculateRMSE Estimations and ground_truth have different size!\n";
	}
	return RMSE;
}