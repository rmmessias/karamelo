/* ----------------------------------------------------------------------
 *
 *                    ***       Karamelo       ***
 *               Parallel Material Point Method Simulator
 * 
 * Copyright (2019) Alban de Vaucorbeil, alban.devaucorbeil@monash.edu
 * Materials Science and Engineering, Monash University
 * Clayton VIC 3800, Australia

 * This software is distributed under the GNU General Public License.
 *
 * ----------------------------------------------------------------------- */

#include <dump_particle.h>
#include <domain.h>
#include <error.h>
#include <method.h>
#include <mpm_math.h>
#include <output.h>
#include <solid.h>
#include <universe.h>
#include <update.h>
#include <matrix.h>
#include <iostream>
#include <algorithm>
#include <Kokkos_Core.hpp>

using namespace std;
using namespace MPM_Math;

DumpParticle::DumpParticle(MPM *mpm, vector<string> args) : Dump(mpm, args) {
  // cout << "In DumpParticle::DumpParticle()" << endl;
  for (int i = 5; i < args.size(); i++) {
    if (find(known_var.begin(), known_var.end(), args[i]) != known_var.end()) {
      output_var.push_back(args[i]);
    } else {
      string error_str = "Error: output variable \033[1;31m" + args[i] +
                         "\033[0m is unknown!\n";
      error_str += "Availabe output variables: ";
      for (auto v : known_var) {
        error_str += v + ", ";
      }
      error->all(FLERR, error_str);
    }
  }
}

DumpParticle::~DumpParticle()
{
}


void DumpParticle::write()
{
  // Open dump file:
  size_t pos_asterisk = filename.find('*');
  string fdump;

  if (pos_asterisk != string::npos)
    {
      // Replace the asterisk by proc-N.ntimestep:
      fdump = filename.substr(0, pos_asterisk);
      if (universe->nprocs > 1) {
	fdump += "proc-" + to_string(universe->me) + ".";
      }
      fdump += to_string(update->ntimestep);
      if (filename.size()-pos_asterisk-1 > 0)
	fdump += filename.substr(pos_asterisk+1, filename.size()-pos_asterisk-1);
    }
  else fdump = filename;

  // cout << "Filemame for dump: " << fdump << endl;

  // Open the file fdump:
  ofstream dumpstream(fdump, ios_base::out);

  if (dumpstream.is_open()) {
    dumpstream << "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n";

    bigint total_np = 0;
    for (int isolid=0; isolid < domain->solids.size(); isolid++) total_np += domain->solids[isolid]->np_local;

    dumpstream << total_np << endl;
    dumpstream << "ITEM: BOX BOUNDS sm sm sm\n";
    dumpstream << domain->boxlo[0] << " " << domain->boxhi[0] << endl;
    dumpstream << domain->boxlo[1] << " " << domain->boxhi[1] << endl;
    dumpstream << domain->boxlo[2] << " " << domain->boxhi[2] << endl;
    dumpstream << "ITEM: ATOMS id type tag ";
    for (auto v: output_var) {
      dumpstream << v << " ";
    }
    dumpstream << endl;

    Matrix3d sigma_;

    for (int isolid=0; isolid < domain->solids.size(); isolid++) {
      Solid *s = domain->solids[isolid];
      
      auto h_ptag = Kokkos::create_mirror_view(s->ptag);
      auto h_x = Kokkos::create_mirror_view(s->x);
      auto h_x0 = Kokkos::create_mirror_view(s->x0);
      auto h_v = Kokkos::create_mirror_view(s->v);
      auto h_sigma = Kokkos::create_mirror_view(s->sigma);
      auto h_strain_el = Kokkos::create_mirror_view(s->strain_el);
      auto h_mbp = Kokkos::create_mirror_view(s->mbp);
      auto h_damage = Kokkos::create_mirror_view(s->damage);
      auto h_damage_init = Kokkos::create_mirror_view(s->damage_init);
      auto h_vol = Kokkos::create_mirror_view(s->vol);
      auto h_mass = Kokkos::create_mirror_view(s->mass);
      auto h_eff_plastic_strain = Kokkos::create_mirror_view(s->eff_plastic_strain);
      auto h_eff_plastic_strain_rate = Kokkos::create_mirror_view(s->eff_plastic_strain_rate);
      auto h_ienergy = Kokkos::create_mirror_view(s->ienergy);
      auto h_R = Kokkos::create_mirror_view(s->R);
      
      Kokkos::deep_copy(h_ptag, s->ptag);
      Kokkos::deep_copy(h_x, s->x);
      Kokkos::deep_copy(h_x0, s->x0);
      Kokkos::deep_copy(h_v, s->v);
      Kokkos::deep_copy(h_sigma, s->sigma);
      Kokkos::deep_copy(h_strain_el, s->strain_el);
      Kokkos::deep_copy(h_mbp, s->mbp);
      Kokkos::deep_copy(h_damage, s->damage);
      Kokkos::deep_copy(h_damage_init, s->damage_init);
      Kokkos::deep_copy(h_vol, s->vol);
      Kokkos::deep_copy(h_mass, s->mass);
      Kokkos::deep_copy(h_eff_plastic_strain, s->eff_plastic_strain);
      Kokkos::deep_copy(h_eff_plastic_strain_rate, s->eff_plastic_strain_rate);
      Kokkos::deep_copy(h_ienergy, s->ienergy);
      
      if (update->method_type == "tlmpm" || update->method_type == "tlcpdi") {
        Kokkos::deep_copy(h_R, s->R);
      }
      
      Kokkos::View<double*>::HostMirror h_T;
      Kokkos::View<double*>::HostMirror h_gamma;
      if (update->method->temp) {
        h_T = Kokkos::create_mirror_view(s->T);
        h_gamma = Kokkos::create_mirror_view(s->gamma);
        Kokkos::deep_copy(h_T, s->T);
        Kokkos::deep_copy(h_gamma, s->gamma);
      }
      
      for (bigint i=0; i<s->np_local;i++) {
        if (update->method_type == "tlmpm" ||
            update->method_type == "tlcpdi")
          sigma_ = h_R[i] * h_sigma[i] * h_R[i].transpose();
        else
          sigma_ = h_sigma[i];
        dumpstream << h_ptag[i] << " ";
        dumpstream << isolid+1 << " ";
        dumpstream << h_ptag[i] << " ";
        for (auto v: output_var) {
          if (v.compare("x")==0) dumpstream << h_x[i][0] << " ";
          else if (v.compare("y")==0) dumpstream << h_x[i][1] << " ";
          else if (v.compare("z")==0) dumpstream << h_x[i][2] << " ";
          else if (v.compare("x0")==0) dumpstream << h_x0[i][0] << " ";
          else if (v.compare("y0")==0) dumpstream << h_x0[i][1] << " ";
          else if (v.compare("z0")==0) dumpstream << h_x0[i][2] << " ";
          else if (v.compare("vx")==0) dumpstream << h_v[i][0] << " ";
          else if (v.compare("vy")==0) dumpstream << h_v[i][1] << " ";
          else if (v.compare("vz")==0) dumpstream << h_v[i][2] << " ";
          else if (v.compare("s11")==0) dumpstream << sigma_(0,0) << " ";
          else if (v.compare("s22")==0) dumpstream << sigma_(1,1) << " ";
          else if (v.compare("s33")==0) dumpstream << sigma_(2,2) << " ";
          else if (v.compare("s12")==0) dumpstream << sigma_(0,1) << " ";
          else if (v.compare("s13")==0) dumpstream << sigma_(0,2) << " ";
          else if (v.compare("s23")==0) dumpstream << sigma_(1,2) << " ";
          else if (v.compare("seq")==0) dumpstream << sqrt(3. / 2.) * Deviator(sigma_).norm() << " ";
          else if (v.compare("e11")==0) dumpstream << h_strain_el[i](0,0) << " ";
          else if (v.compare("e22")==0) dumpstream << h_strain_el[i](1,1) << " ";
          else if (v.compare("e33")==0) dumpstream << h_strain_el[i](2,2) << " ";
          else if (v.compare("e12")==0) dumpstream << h_strain_el[i](0,1) << " ";
          else if (v.compare("e13")==0) dumpstream << h_strain_el[i](0,2) << " ";
          else if (v.compare("e23")==0) dumpstream << h_strain_el[i](1,2) << " ";
          else if (v.compare("damage")==0) dumpstream << h_damage[i] << " ";
          else if (v.compare("damage_init")==0) dumpstream << h_damage_init[i] << " ";
          else if (v.compare("volume")==0) dumpstream << h_vol[i] << " ";
          else if (v.compare("mass")==0) dumpstream << h_mass[i] << " ";
          else if (v.compare("bx")==0) dumpstream << h_mbp[i][0] << " ";
          else if (v.compare("by")==0) dumpstream << h_mbp[i][1] << " ";
          else if (v.compare("bz")==0) dumpstream << h_mbp[i][2] << " ";
          else if (v.compare("ep")==0) dumpstream << h_eff_plastic_strain[i] << " ";
          else if (v.compare("epdot")==0) dumpstream << h_eff_plastic_strain_rate[i] << " ";
          else if (v.compare("ienergy")==0) dumpstream << h_ienergy[i] << " ";
          else if (v.compare("T")==0) {
            if (update->method->temp) {
              dumpstream << h_T[i] << " ";
            } else {
              dumpstream << "0 ";
            }
          }
          else if (v.compare("gamma")==0) {
            if (update->method->temp) {
              dumpstream << h_gamma[i] << " ";
            } else {
              dumpstream << "0 ";	      
            }
          }
        }
        dumpstream << endl;
      }
    }
    dumpstream.close();
  } else {
    error->all(FLERR, "Error: cannot write in file: " + fdump + ".\n");
  }
}
