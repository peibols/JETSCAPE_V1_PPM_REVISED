#include <omp.h>
#include <algorithm>
#include <memory>
#include <cmath>

#include "PPMmusic_freezeout.h"
#include "JetScapeLogger.h"
#include "./cornelius.h"

using namespace Jetscape;
using namespace std;

#ifndef _OPENMP
  #define omp_get_thread_num() 0
#endif

// Cornelius freeze out (C. Shen, 11/2014)
void Freezeout::IsochronousFreezeout() {
    // this function freeze-out fluid cells between epsFO and epsFO_low
    // on an equal time hyper-surface at the first time step
    // this function will be trigged if freezeout_lowtemp_flag == 1
    const int neta = arena.nEta();
    const int fac_eta = 1;
   
    //double epsFO = epsFO_list[i_freezesurf]/hbarc;
    double epsFO = eos->E(temp_fo);
    #pragma omp parallel for
    for (int ieta = 3; ieta < neta - fac_eta - 3; ieta += fac_eta) {
      int thread_id = omp_get_thread_num();
      FreezeOut_equal_tau_Surface_XY(ieta,
                                     thread_id, epsFO);
    }
}

int Freezeout::FullFreezeout(SCGrid &arena_freezeout) {
    const int neta = arena.nEta();
    const int fac_eta = 1;
   
    int intersections = 0; 
    //double epsFO = epsFO_list[i_freezesurf]/hbarc;
    double epsFO = eos->E(temp_fo);
    #pragma omp parallel for reduction(+:intersections)
    for (int ieta = 3; ieta < neta - fac_eta - 3; ieta += fac_eta) {
      int thread_id = omp_get_thread_num();
      intersections += FindFreezeOutSurface_Cornelius_XY(
		      ieta, arena_freezeout, thread_id, epsFO);
    }
    if (intersections == 0) {
        std::cout << "All cells frozen out. Exiting." << std::endl;
        return ppm_finished;
    }
    return ppm_running;

}

int Freezeout::FindFreezeOutSurface_Cornelius_XY(int ieta,
                                              SCGrid &arena_freezeout,
                                              int thread_id, double epsFO) {
    double tau = coord->tau*hbarc;
    const int nx = arena.nX();
    const int ny = arena.nY();

    stringstream strs_name;
    strs_name << "surface_temp_" << setprecision(4) << temp_fo
              << "_" << thread_id << ".dat";
    ofstream s_file;
    s_file.open(strs_name.str().c_str(), ios::out | ios::app);
    
    const int dim = 4;
    int intersections = 0;

    //facTau      = DATA.facTau;   // step to skip in tau direction
    //int fac_x   = DATA.fac_x;
    //int fac_y   = DATA.fac_y;
    int facTau      = 1;
    int fac_x   = 1;
    int fac_y   = 1;
    int fac_eta = 1;

    const double DTAU = facTau*DATA.delta_tau;
    const double DX   = fac_x*DATA.delta_x;
    const double DY   = fac_y*DATA.delta_y;
    const double DETA = fac_eta*DATA.delta_eta;

    // initialize Cornelius
    double lattice_spacing[4] = {DTAU, DX, DY, DETA};
    std::shared_ptr<Cornelius> cornelius_ptr(new Cornelius());
    cornelius_ptr->init(dim, epsFO/pow(hbarc,4.), lattice_spacing);

    // initialize the hyper-cube for Cornelius
    double ****cube = new double*** [2];
    for (int i = 0; i < 2; i++) {
        cube[i] = new double** [2];
        for (int j = 0; j < 2; j++) {
            cube[i][j] = new double* [2];
            for (int k = 0; k < 2; k++) {
                cube[i][j][k] = new double[2];
                for (int l = 0; l < 2; l++)
                    cube[i][j][k][l] = 0.0;
            }
        }
    }

    double x_fraction[2][4];
    double eta = (DATA.delta_eta)*ieta - (DATA.eta_size)/2.0;
    for (int ix = 3; ix < nx - fac_x - 3; ix += fac_x) {
        double x = ix*(DATA.delta_x) - (DATA.x_size/2.0);
        for (int iy = 3; iy < ny - fac_y - 3; iy += fac_y) {
            double y = iy*(DATA.delta_y) - (DATA.y_size/2.0);

            // judge intersection (from Bjoern)
            int intersect = 1;
            if ((arena(ix+fac_x,iy+fac_y,ieta+fac_eta).epsilon-epsFO)
                *(arena_freezeout(ix,iy,ieta).epsilon-epsFO)>0.)
                if((arena(ix+fac_x,iy,ieta).epsilon-epsFO)
                    *(arena_freezeout(ix,iy+fac_y,ieta+fac_eta).epsilon-epsFO)>0.)
                    if((arena(ix,iy+fac_y,ieta).epsilon-epsFO)
                        *(arena_freezeout(ix+fac_x,iy,ieta+fac_eta).epsilon-epsFO)>0.)
                        if((arena(ix,iy,ieta+fac_eta).epsilon-epsFO)
                            *(arena_freezeout(ix+fac_x,iy+fac_y,ieta).epsilon-epsFO)>0.)
                            if((arena(ix+fac_x,iy+fac_y,ieta).epsilon-epsFO)
                                *(arena_freezeout(ix,iy,ieta+fac_eta).epsilon-epsFO)>0.)
                                if((arena(ix+fac_x,iy,ieta+fac_eta).epsilon-epsFO)
                                    *(arena_freezeout(ix,iy+fac_y,ieta).epsilon-epsFO)>0.)
                                    if((arena(ix,iy+fac_y,ieta+fac_eta).epsilon-epsFO)
                                        *(arena_freezeout(ix+fac_x,iy,ieta).epsilon-epsFO)>0.)
                                        if((arena(ix,iy,ieta).epsilon-epsFO)
                                            *(arena_freezeout(ix+fac_x,iy+fac_y,ieta+fac_eta).epsilon-epsFO)>0.)
                                                intersect=0;

            if (intersect==0) continue;

            // if intersect, prepare for the hyper-cube
            intersections++;
            cube[0][0][0][0] = pow(hbarc,-4.)*arena_freezeout(ix      , iy      , ieta        ).epsilon;
            cube[0][0][1][0] = pow(hbarc,-4.)*arena_freezeout(ix      , iy+fac_y, ieta        ).epsilon;
            cube[0][1][0][0] = pow(hbarc,-4.)*arena_freezeout(ix+fac_x, iy      , ieta        ).epsilon;
            cube[0][1][1][0] = pow(hbarc,-4.)*arena_freezeout(ix+fac_x, iy+fac_y, ieta        ).epsilon;
            cube[1][0][0][0] = pow(hbarc,-4.)*arena  (ix      , iy      , ieta        ).epsilon;
            cube[1][0][1][0] = pow(hbarc,-4.)*arena  (ix      , iy+fac_y, ieta        ).epsilon;
            cube[1][1][0][0] = pow(hbarc,-4.)*arena  (ix+fac_x, iy      , ieta        ).epsilon;
            cube[1][1][1][0] = pow(hbarc,-4.)*arena  (ix+fac_x, iy+fac_y, ieta        ).epsilon;
            cube[0][0][0][1] = pow(hbarc,-4.)*arena_freezeout(ix      , iy      , ieta+fac_eta).epsilon;
            cube[0][0][1][1] = pow(hbarc,-4.)*arena_freezeout(ix      , iy+fac_y, ieta+fac_eta).epsilon;
            cube[0][1][0][1] = pow(hbarc,-4.)*arena_freezeout(ix+fac_x, iy      , ieta+fac_eta).epsilon;
            cube[0][1][1][1] = pow(hbarc,-4.)*arena_freezeout(ix+fac_x, iy+fac_y, ieta+fac_eta).epsilon;
            cube[1][0][0][1] = pow(hbarc,-4.)*arena  (ix      , iy      , ieta+fac_eta).epsilon;
            cube[1][0][1][1] = pow(hbarc,-4.)*arena  (ix      , iy+fac_y, ieta+fac_eta).epsilon;
            cube[1][1][0][1] = pow(hbarc,-4.)*arena  (ix+fac_x, iy      , ieta+fac_eta).epsilon;
            cube[1][1][1][1] = pow(hbarc,-4.)*arena  (ix+fac_x, iy+fac_y, ieta+fac_eta).epsilon;

    
            // Now, the magic will happen in the Cornelius ...
            cornelius_ptr->find_surface_4d(cube);

            // get positions of the freeze-out surface
            // and interpolating results
            for (int isurf = 0; isurf < cornelius_ptr->get_Nelements();
                 isurf++) {
                // surface normal vector d^3 \sigma_\mu
                double FULLSU[4];
                for (int ii = 0; ii < 4; ii++)
                    FULLSU[ii] = cornelius_ptr->get_normal_elem(isurf, ii);

                // check the size of the surface normal vector
                if (std::abs(FULLSU[0]) > (DX*DY*DETA+0.01)) {
                    JSINFO << "problem: volume in tau direction "
                                  << std::abs(FULLSU[0]) << "  > DX*DY*DETA = "
                                  << DX*DY*DETA;
                }
                if (std::abs(FULLSU[1]) > (DTAU*DY*DETA+0.01)) {
                    JSINFO << "problem: volume in x direction "
                                  << std::abs(FULLSU[1])
                                  << "  > DTAU*DY*DETA = " << DTAU*DY*DETA;
                }
                if (std::abs(FULLSU[2]) > (DX*DTAU*DETA+0.01)) {
                    JSINFO << "problem: volume in y direction "
                                  << std::abs(FULLSU[2])
                                  << "  > DX*DTAU*DETA = " << DX*DTAU*DETA;
                }
                if (std::abs(FULLSU[3]) > (DX*DY*DTAU+0.01)) {
                    JSINFO << "problem: volume in eta direction "
                                  << std::abs(FULLSU[3]) << "  > DX*DY*DTAU = "
                                  << DX*DY*DTAU;
                }

                // position of the freeze-out fluid cell
                for (int ii = 0; ii < 4; ii++) {
                    x_fraction[1][ii] =
                        cornelius_ptr->get_centroid_elem(isurf, ii);
                    x_fraction[0][ii] =
                        lattice_spacing[ii] - x_fraction[1][ii];
                }
                const double tau_center = tau - DTAU + x_fraction[1][0];
                const double x_center = x + x_fraction[1][1];
                const double y_center = y + x_fraction[1][2];
                const double eta_center = eta + x_fraction[1][3];

                // perform 4-d linear interpolation for all fluid
                // quantities

                // flow velocity u^x
                cube[0][0][0][0] = arena_freezeout(ix      , iy      , ieta        ).u[1];
                cube[0][0][1][0] = arena_freezeout(ix      , iy+fac_y, ieta        ).u[1];
                cube[0][1][0][0] = arena_freezeout(ix+fac_x, iy      , ieta        ).u[1];
                cube[0][1][1][0] = arena_freezeout(ix+fac_x, iy+fac_y, ieta        ).u[1];
                cube[1][0][0][0] = arena  (ix      , iy      , ieta        ).u[1];
                cube[1][0][1][0] = arena  (ix      , iy+fac_y, ieta        ).u[1];
                cube[1][1][0][0] = arena  (ix+fac_x, iy      , ieta        ).u[1];
                cube[1][1][1][0] = arena  (ix+fac_x, iy+fac_y, ieta        ).u[1];
                cube[0][0][0][1] = arena_freezeout(ix      , iy      , ieta+fac_eta).u[1];
                cube[0][0][1][1] = arena_freezeout(ix      , iy+fac_y, ieta+fac_eta).u[1];
                cube[0][1][0][1] = arena_freezeout(ix+fac_x, iy      , ieta+fac_eta).u[1];
                cube[0][1][1][1] = arena_freezeout(ix+fac_x, iy+fac_y, ieta+fac_eta).u[1];
                cube[1][0][0][1] = arena  (ix      , iy      , ieta+fac_eta).u[1];
                cube[1][0][1][1] = arena  (ix      , iy+fac_y, ieta+fac_eta).u[1];
                cube[1][1][0][1] = arena  (ix+fac_x, iy      , ieta+fac_eta).u[1];
                cube[1][1][1][1] = arena  (ix+fac_x, iy+fac_y, ieta+fac_eta).u[1];
                const double ux_center = 
                    Util::four_dimension_linear_interpolation(
                                lattice_spacing, x_fraction, cube);

                // flow velocity u^y
                cube[0][0][0][0] = arena_freezeout(ix      , iy      , ieta        ).u[2];
                cube[0][0][1][0] = arena_freezeout(ix      , iy+fac_y, ieta        ).u[2];
                cube[0][1][0][0] = arena_freezeout(ix+fac_x, iy      , ieta        ).u[2];
                cube[0][1][1][0] = arena_freezeout(ix+fac_x, iy+fac_y, ieta        ).u[2];
                cube[1][0][0][0] = arena  (ix      , iy      , ieta        ).u[2];
                cube[1][0][1][0] = arena  (ix      , iy+fac_y, ieta        ).u[2];
                cube[1][1][0][0] = arena  (ix+fac_x, iy      , ieta        ).u[2];
                cube[1][1][1][0] = arena  (ix+fac_x, iy+fac_y, ieta        ).u[2];
                cube[0][0][0][1] = arena_freezeout(ix      , iy      , ieta+fac_eta).u[2];
                cube[0][0][1][1] = arena_freezeout(ix      , iy+fac_y, ieta+fac_eta).u[2];
                cube[0][1][0][1] = arena_freezeout(ix+fac_x, iy      , ieta+fac_eta).u[2];
                cube[0][1][1][1] = arena_freezeout(ix+fac_x, iy+fac_y, ieta+fac_eta).u[2];
                cube[1][0][0][1] = arena  (ix      , iy      , ieta+fac_eta).u[2];
                cube[1][0][1][1] = arena  (ix      , iy+fac_y, ieta+fac_eta).u[2];
                cube[1][1][0][1] = arena  (ix+fac_x, iy      , ieta+fac_eta).u[2];
                cube[1][1][1][1] = arena  (ix+fac_x, iy+fac_y, ieta+fac_eta).u[2];
                const double uy_center = 
                    Util::four_dimension_linear_interpolation(
                                lattice_spacing, x_fraction, cube);

                // flow velocity u^eta
                cube[0][0][0][0] = arena_freezeout(ix      , iy      , ieta        ).u[3];
                cube[0][0][1][0] = arena_freezeout(ix      , iy+fac_y, ieta        ).u[3];
                cube[0][1][0][0] = arena_freezeout(ix+fac_x, iy      , ieta        ).u[3];
                cube[0][1][1][0] = arena_freezeout(ix+fac_x, iy+fac_y, ieta        ).u[3];
                cube[1][0][0][0] = arena  (ix      , iy      , ieta        ).u[3];
                cube[1][0][1][0] = arena  (ix      , iy+fac_y, ieta        ).u[3];
                cube[1][1][0][0] = arena  (ix+fac_x, iy      , ieta        ).u[3];
                cube[1][1][1][0] = arena  (ix+fac_x, iy+fac_y, ieta        ).u[3];
                cube[0][0][0][1] = arena_freezeout(ix      , iy      , ieta+fac_eta).u[3];
                cube[0][0][1][1] = arena_freezeout(ix      , iy+fac_y, ieta+fac_eta).u[3];
                cube[0][1][0][1] = arena_freezeout(ix+fac_x, iy      , ieta+fac_eta).u[3];
                cube[0][1][1][1] = arena_freezeout(ix+fac_x, iy+fac_y, ieta+fac_eta).u[3];
                cube[1][0][0][1] = arena  (ix      , iy      , ieta+fac_eta).u[3];
                cube[1][0][1][1] = arena  (ix      , iy+fac_y, ieta+fac_eta).u[3];
                cube[1][1][0][1] = arena  (ix+fac_x, iy      , ieta+fac_eta).u[3];
                cube[1][1][1][1] = arena  (ix+fac_x, iy+fac_y, ieta+fac_eta).u[3];
                const double ueta_center = 
                    Util::four_dimension_linear_interpolation(
                                lattice_spacing, x_fraction, cube);

                // reconstruct u^tau from u^i
                const double utau_center = sqrt(1. + ux_center*ux_center 
                                   + uy_center*uy_center 
                                   + ueta_center*ueta_center);
                
		// 4-dimension interpolation done
                const double TFO = temp_fo;
                const double muB = 0.;

                double pressure = eos->P(epsFO); 
                const double eps_plus_p_over_T_FO = (epsFO + pressure)/TFO;

                // finally output results !!!!
                    s_file << scientific << setprecision(10)
                           << tau_center << " " << x_center << " "
                           << y_center << " " << eta_center << " "
                           << FULLSU[0] << " " << FULLSU[1] << " "
                           << FULLSU[2] << " " << FULLSU[3] << " "
                           << utau_center << " " << ux_center << " "
                           << uy_center << " " << ueta_center << " "
                           << epsFO << " " << TFO << " " << muB << " "
                           << eps_plus_p_over_T_FO << " "
                           << 0. << " " << 0. << " "
                           << 0. << " " << 0. << " "
                           << 0. << " " << 0. << " "
                           << 0. << " " << 0. << " "
                           << 0. << " " << 0. << " ";
		    s_file << endl;
            }
        }
    }
    s_file.close();

    // clean up
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++)
                delete [] cube[i][j][k];
            delete [] cube[i][j];
        }
        delete [] cube[i];
    }
    delete [] cube;
    return(intersections);

}   
    
 void Freezeout::FreezeOut_equal_tau_Surface_XY(int ieta,
                                               int thread_id, double epsFO) {
    double tau = coord->tau*hbarc;
	
    //const bool surface_in_binary = DATA.freeze_surface_in_binary;
    double epsFO_low = 0.05/hbarc;        // 1/fm^4

    //convert to GeV^4
    epsFO_low*=pow(hbarc,4.);

    const int nx = arena.nX();
    const int ny = arena.nY();

    stringstream strs_name;
    strs_name << "surface_temp_" << setprecision(4) << temp_fo
                  << "_" << thread_id << ".dat";
    ofstream s_file;
    //if (surface_in_binary) {
      //  s_file.open(strs_name.str().c_str(),
        //            ios::out | ios::app | ios::binary);
    //} else {
        s_file.open(strs_name.str().c_str(), ios::out | ios::app);
    //}

  //  const int fac_x   = DATA.fac_x;
  //  const int fac_y   = DATA.fac_y;
    const int fac_eta = 1;
    const int fac_x   = 1;
    const int fac_y   = 1;

    const double DX   = fac_x*DATA.delta_x;
    const double DY   = fac_y*DATA.delta_y;
    const double DETA = fac_eta*DATA.delta_eta;

    double eta = (DATA.delta_eta)*ieta - (DATA.eta_size)/2.0;
    for (int ix = 3; ix < nx - fac_x - 3; ix += fac_x) {
        double x = ix*(DATA.delta_x) - (DATA.x_size/2.0); 
        for (int iy = 3; iy < ny - fac_y - 3; iy += fac_y) {
            double y = iy*(DATA.delta_y) - (DATA.y_size/2.0);

            // judge intersection
            if (arena(ix,iy,ieta).epsilon > epsFO) continue;
            if (arena(ix,iy,ieta).epsilon < epsFO_low) continue;

            // surface normal vector d^3 \sigma_\mu
            const double FULLSU[] = {DX*DY*DETA, 0.0, 0.0, 0.0};

            // get positions of the freeze-out surface
            const double tau_center = tau;
            const double x_center   = x;
            const double y_center   = y;
            const double eta_center = eta;

            // flow velocity
            const double ux_center   = arena(ix, iy, ieta).u[1];
            const double uy_center   = arena(ix, iy, ieta).u[2];
            const double ueta_center = arena(ix, iy, ieta).u[3];  // u^eta/tau
            // reconstruct u^tau from u^i
            const double utau_center = sqrt(1. + ux_center*ux_center 
                                               + uy_center*uy_center 
                                               + ueta_center*ueta_center);

	    /*
            // baryon density rho_b
            const double rhob_center = arena(ix, iy, ieta).rhob;


            // baryon diffusion current
            double qtau_center = arena(ix, iy, ieta).Wmunu[10];
            double qx_center   = arena(ix, iy, ieta).Wmunu[11];
            double qy_center   = arena(ix, iy, ieta).Wmunu[12];
            double qeta_center = arena(ix, iy, ieta).Wmunu[13];

            // reconstruct q^\tau from the transverality criteria
            double u_flow[]       = {utau_center, ux_center, uy_center, ueta_center};
            double q_mu[]         = {qtau_center, qx_center, qy_center, qeta_center};
            double q_regulated[4] = {0.0, 0.0, 0.0, 0.0};
            regulate_qmu(u_flow, q_mu, q_regulated);
            qtau_center = q_regulated[0];
            qx_center   = q_regulated[1];
            qy_center   = q_regulated[2];
            qeta_center = q_regulated[3];

            // bulk viscous pressure pi_b
            const double pi_b_center = arena(ix,iy,ieta).pi_b;

            // shear viscous tensor
            double Wtautau_center = arena(ix, iy, ieta).Wmunu[0];
            double Wtaux_center   = arena(ix, iy, ieta).Wmunu[1];
            double Wtauy_center   = arena(ix, iy, ieta).Wmunu[2];
            double Wtaueta_center = arena(ix, iy, ieta).Wmunu[3];
            double Wxx_center     = arena(ix, iy, ieta).Wmunu[4];
            double Wxy_center     = arena(ix, iy, ieta).Wmunu[5];
            double Wxeta_center   = arena(ix, iy, ieta).Wmunu[6];
            double Wyy_center     = arena(ix, iy, ieta).Wmunu[7];
            double Wyeta_center   = arena(ix, iy, ieta).Wmunu[8];
            double Wetaeta_center = arena(ix, iy, ieta).Wmunu[9];
            // regulate Wmunu according to transversality and traceless
            double Wmunu_input[4][4];
            double Wmunu_regulated[4][4];
            Wmunu_input[0][0] = Wtautau_center;
            Wmunu_input[0][1] = Wmunu_input[1][0] = Wtaux_center;
            Wmunu_input[0][2] = Wmunu_input[2][0] = Wtauy_center;
            Wmunu_input[0][3] = Wmunu_input[3][0] = Wtaueta_center;
            Wmunu_input[1][1] = Wxx_center;
            Wmunu_input[1][2] = Wmunu_input[2][1] = Wxy_center;
            Wmunu_input[1][3] = Wmunu_input[3][1] = Wxeta_center;
            Wmunu_input[2][2] = Wyy_center;
            Wmunu_input[2][3] = Wmunu_input[3][2] = Wyeta_center;
            Wmunu_input[3][3] = Wetaeta_center;
            regulate_Wmunu(u_flow, Wmunu_input, Wmunu_regulated);
            Wtautau_center = Wmunu_regulated[0][0];
            Wtaux_center   = Wmunu_regulated[0][1];
            Wtauy_center   = Wmunu_regulated[0][2];
            Wtaueta_center = Wmunu_regulated[0][3];
            Wxx_center     = Wmunu_regulated[1][1];
            Wxy_center     = Wmunu_regulated[1][2];
            Wxeta_center   = Wmunu_regulated[1][3];
            Wyy_center     = Wmunu_regulated[2][2];
            Wyeta_center   = Wmunu_regulated[2][3];
            Wetaeta_center = Wmunu_regulated[3][3];
*/
            // get other thermodynamical quantities
            double e_local   = arena(ix, iy, ieta).epsilon;
            double T_local   = arena(ix, iy, ieta).T;
	    //double T_local   = eos.get_temperature(e_local, rhob_center);
            //double muB_local = eos.get_mu(e_local, rhob_center);
 	    double muB_local = 0.;
 	    if (T_local < 0) {
		JSINFO << "Evolve::FreezeOut_equal_tau_Surface: "
                              << "T_local = " << T_local
                              << " <0. ERROR. exiting.";
                //JSINFO.flush("error");
                exit(1);
            }

            //double pressure = eos.get_pressure(e_local, rhob_center);
            double pressure = arena(ix, iy, ieta).p;
	    double eps_plus_p_over_T = (e_local + pressure)/T_local;
/*
            // finally output results
            if (surface_in_binary) {
                float array[] = {static_cast<float>(tau_center),
                                 static_cast<float>(x_center),
                                 static_cast<float>(y_center),
                                 static_cast<float>(eta_center),
                                 static_cast<float>(FULLSU[0]),
                                 static_cast<float>(FULLSU[1]),
                                 static_cast<float>(FULLSU[2]),
                                 static_cast<float>(FULLSU[3]),
                                 static_cast<float>(utau_center),
                                 static_cast<float>(ux_center),
                                 static_cast<float>(uy_center),
                                 static_cast<float>(ueta_center),
                                 static_cast<float>(e_local),
                                 static_cast<float>(T_local),
                                 static_cast<float>(muB_local),
                                 static_cast<float>(eps_plus_p_over_T),
                                 static_cast<float>(Wtautau_center),
                                 static_cast<float>(Wtaux_center),
                                 static_cast<float>(Wtauy_center),
                                 static_cast<float>(Wtaueta_center),
                                 static_cast<float>(Wxx_center),
                                 static_cast<float>(Wxy_center),
                                 static_cast<float>(Wxeta_center),
                                 static_cast<float>(Wyy_center),
                                 static_cast<float>(Wyeta_center),
                                 static_cast<float>(Wetaeta_center),
                                 static_cast<float>(pi_b_center),
                                 static_cast<float>(rhob_center),
                                 static_cast<float>(qtau_center),
                                 static_cast<float>(qx_center),
                                 static_cast<float>(qy_center),
                                 static_cast<float>(qeta_center)};
                for (int i = 0; i < 32; i++) {
                    s_file.write((char*) &(array[i]), sizeof(float));
                }
            } else {
*/
    	      s_file << scientific << setprecision(10) 
                       << tau_center     << " " << x_center          << " " 
                       << y_center       << " " << eta_center        << " " 
                       << FULLSU[0]      << " " << FULLSU[1]         << " " 
                       << FULLSU[2]      << " " << FULLSU[3]         << " " 
                       << utau_center    << " " << ux_center         << " " 
                       << uy_center      << " " << ueta_center       << " " 
                       << e_local        << " " << T_local           << " "
                       << muB_local      << " " << eps_plus_p_over_T << " " 
                       //<< Wtautau_center << " " << Wtaux_center      << " " 
                       //<< Wtauy_center   << " " << Wtaueta_center    << " " 
                       //<< Wxx_center     << " " << Wxy_center        << " " 
                       //<< Wxeta_center   << " " << Wyy_center        << " "
                       //<< Wyeta_center   << " " << Wetaeta_center    << " " ;
                       << 0. << " " << 0. << " "  
                       << 0. << " " << 0. << " "  
                       << 0. << " " << 0. << " "  
                       << 0. << " " << 0. << " "  
                       << 0. << " " << 0. << " ";  
	      s_file << endl;
//            }
        }
    }
    s_file.close();
}

