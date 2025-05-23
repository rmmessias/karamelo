/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef DUMP_CLASS

DumpStyle(grid/gz,DumpGridGz)

#else

#ifndef MPM_DUMP_GRID_GZ_H
#define MPM_DUMP_GRID_GZ_H

#include <deque>
#include <dump.h>
#include <grid.h>
#include <thread>
#include <utility>

class DumpGridGz : public Dump {
  Kokkos::View<tagint*,Kokkos::SharedSpace>::HostMirror ntag;   ///< unique identifier for nodes in the system.
  
  Kokkos::View<Vector3d*,Kokkos::SharedSpace>::HostMirror x;            ///< nodes' current position
  Kokkos::View<Vector3d**,Kokkos::SharedSpace>::HostMirror v;           ///< nodes' velocity at time t
  Kokkos::View<Vector3d**,Kokkos::SharedSpace>::HostMirror v_update;    ///< nodes' velocity at time t + dt
  Kokkos::View<Vector3d**,Kokkos::SharedSpace>::HostMirror mb;          ///< nodes' external forces times the mass

  Kokkos::View<double**,Kokkos::SharedSpace>::HostMirror mass;           ///< nodes' current mass
  Kokkos::View<int*,Kokkos::SharedSpace>::HostMirror mask;              ///< nodes' group mask
  Kokkos::View<double**,Kokkos::SharedSpace>::HostMirror vol;            ///< nodes' current volume
  Kokkos::View<bool**,Kokkos::SharedSpace>::HostMirror rigid;           ///< are the nodes in the area of influence of a rigid body?
  Kokkos::View<Vector3i*,Kokkos::SharedSpace>::HostMirror ntype;        ///< node type in x, y, and z directions (False for an edge, True otherwise)

  Kokkos::View<double**,Kokkos::SharedSpace>::HostMirror T;              ///< nodes' temperature at time t

 public:
  DumpGridGz(MPM *, vector<string>);
  ~DumpGridGz();

  void write();
 protected:
  vector<string> known_var = {"x", "y", "z",
			      "vx", "vy", "vz",
			      "vx_update", "vy_update", "vz_update",
			      "bx", "by", "bz",
			      "mass", "mask", "vol",
			      "rigid", "T",
			      "ntypex", "ntypey", "ntypez"};
private:
  deque<pair<thread, vector<double>>>  threads;        ///< Pair storing the threads and the buffer

  void write_to_file(bigint, string, bigint, bigint);
};

#endif
#endif
