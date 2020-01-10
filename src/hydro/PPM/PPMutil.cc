#include "PPMutil.h"
#include <iostream>

using namespace std;

namespace Util {

double four_dimension_linear_interpolation(
            double* lattice_spacing, double fraction[2][4], double**** cube) {
    double denorm = 1.0;
    double results = 0.0;
    for (int i = 0; i < 4; i++) {
        denorm *= lattice_spacing[i];
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    results += (cube[i][j][k][l]*fraction[i][0]*fraction[j][1]
                                *fraction[k][2]*fraction[l][3]);
                }
            }
        }
    }
    results = results/denorm;
    return (results);
}


}
