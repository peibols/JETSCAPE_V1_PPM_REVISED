#ifndef UTIL_H
#define UTIL_H

#include <math.h>

#ifndef PI
#define PI (3.14159265358979324)
#endif

#ifndef hbarc
#define hbarc (0.1973)
#endif

namespace Util {

  double four_dimension_linear_interpolation(
	  double* lattice_spacing, double fraction[2][4], double**** cube);

}  

#endif
