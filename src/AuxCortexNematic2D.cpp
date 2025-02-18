#include <iostream>
#include <mpi.h>
#include <fstream>

/// Trilinos headers
#include <Teuchos_RCP.hpp>

/// hiperlife headers
#include "hl_TypeDefs.h"
#include "hl_GlobalBasisFunctions.h"
#include "hl_StructMeshGenerator.h"
#include "hl_DistributedMesh.h"
#include "hl_FillStructure.h"
#include "hl_DOFsHandler.h"
#include "hl_HiPerProblem.h"
#include "hl_LinearSolver_Direct_Amesos.h"
#include "hl_Tensor.h"
#include "hl_LinearSolver_Iterative_AztecOO.h"
#include "AuxCortexNematic2D.h"

void LS_CortexNematic_2D(Teuchos::RCP<hiperlife::FillStructure> fillStr)
{
    using namespace hiperlife;
    using namespace hiperlife::Tensor;

    //-----------------------------------------------------------
    // [1] INPUT DATA: Extract mesh and problem details
    //-----------------------------------------------------------

    // Extract data for the cortex nematic model
    auto& subFill = (*fillStr)["cortexHand"];
    int nDim = subFill.nDim;      // Number of spatial dimensions
    int pDim = subFill.pDim;      // Polynomial basis dimension
    int eNN  = subFill.eNN;       // Number of element nodes
    int numDOFs = subFill.numDOFs; // Degrees of freedom per element

    // Neighboring element coordinates and degrees of freedom (DOFs)
    tensor<double,2> nborCoords(subFill.nborCoords.data(), eNN, nDim);
    tensor<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);
    tensor<double,2> nborDOFs0(subFill.nborDOFs0.data(), eNN, numDOFs);

    // Basis function values and derivatives
    tensor<double,1> bf(subFill.getDer(0), eNN);
    tensor<double,2> Dbf_l(subFill.getDer(1), eNN, pDim);

    // Extract fill structure for pressure-related computations
    auto& subFill_p = (*fillStr)["dhandP"];
    int nDim_p = subFill_p.nDim;
    int eNN_p  = subFill_p.eNN;

    // Neighboring pressure-related elements and DOFs
    tensor<double,2> nborCoords_p(subFill_p.nborCoords.data(), eNN_p, nDim_p);
    tensor<double,1> nborDOFs_p(subFill_p.nborDOFs.data(), eNN_p);
    tensor<double,1> bf_p(subFill_p.nborBFs(), eNN_p);

    //-----------------------------------------------------------
    // [2] EXTRACT MODEL PARAMETERS FROM USER DEFINED STRUCTURE
    //-----------------------------------------------------------

    double deltat = fillStr->userStr->dparam[0]; // Time step size

    // Susceptibility and critical height parameters
    double sus20  = fillStr->userStr->dparam[1];
    double sus40  = fillStr->userStr->dparam[2];
    double hcrit  = fillStr->userStr->dparam[3];

    // Growth and depolymerization parameters
    double kp  = fillStr->userStr->dparam[4];
    double kd0 = fillStr->userStr->dparam[5];

    // Frank elasticity constant
    double frank  = fillStr->userStr->dparam[6];

    // Contraction parameters
    double lambda_iso    = fillStr->userStr->dparam[7];
    double lambda_aniso  = fillStr->userStr->dparam[8];
    double lambda_rot    = fillStr->userStr->dparam[9];

    // Viscosity parameters
    double visc    = fillStr->userStr->dparam[10];
    double rvisc   = fillStr->userStr->dparam[11];
    double cvisc   = fillStr->userStr->dparam[12];

    // Friction, stabilization, and equilibrium height
    double fric    = fillStr->userStr->dparam[13];
    double stab    = fillStr->userStr->dparam[14];
    double heqb    = fillStr->userStr->dparam[22];

    //-----------------------------------------------------------
    // [3] RESCALE PARAMETERS BASED ON TIME STEP SIZE
    //-----------------------------------------------------------
    sus20  /= deltat;
    sus40  /= deltat;
    frank  /= deltat;

    //-----------------------------------------------------------
    // [4] OUTPUT VARIABLES: System Matrices & Energy Terms
    //-----------------------------------------------------------

    // System matrices for numerical simulation
    tensor<double,2> Bk(fillStr->Bk(0).data(), eNN, numDOFs);
    tensor<double,1> Bk1(fillStr->Bk(1).data(), eNN_p);

    // Hessian matrices for coupled DOFs and pressure terms
    tensor<double,4> Ak(fillStr->Ak(0,0).data(), eNN, numDOFs, eNN, numDOFs);
    tensor<double,3> Ak01(fillStr->Ak(0,1).data(), eNN, numDOFs, eNN_p);
    tensor<double,3> Ak10(fillStr->Ak(1,0).data(), eNN_p, eNN, numDOFs);

    // Energy tracking variables
    double rayleighian{};           // Dissipation function
    double freeEnergyPotential{};    // Free energy contribution
    double dissipation{};            // Dissipation contribution
    double power{};                  // Power potential term

    //-----------------------------------------------------------
    // [5] GEOMETRY PROCESSING: Compute Transformations & Derivatives
    //-----------------------------------------------------------

    // Compute element center coordinates using basis functions
    tensor<double,1> x = bf * nborCoords;

    // Compute transformation matrix (from reference space to spatial coordinates)
    tensor<double,2> T = nborCoords(all, range(0,1)).T() * Dbf_l;

    // Compute Jacobian determinant for integration
    double jac = T.det();

    // Compute global derivatives of basis functions
    tensor<double,2> Dbf_g = Dbf_l * T.inv();

    //-----------------------------------------------------------
    // [6] STATE VARIABLES INITIALIZATION
    //-----------------------------------------------------------

    // [6.1] Thickness-related variables
    tensor<double,1> nbor_h  = nborDOFs(all,0);     // Thickness field h
    tensor<double,1> nbor_h0 = nborDOFs0(all,0);    // Previous time step thickness field h0

    // [6.2] Nematic alignment tensor variables
    tensor<double,2> nbor_q  = nborDOFs(all, range(1,2));  // Q-tensor components
    tensor<double,2> nbor_q0 = nborDOFs0(all, range(1,2)); // Q-tensor from previous step

    // [6.3] Velocity field variables
    tensor<double,2> nbor_v  = nborDOFs(all, range(3,4));  // Velocity components

    // Identity tensor and Voigt notation for tensor operations
    tensor<double,2> id2d = {{1.0, 0.0}, {0.0, 1.0}};
    tensor<double,3> voigt = {{{1,0}, {0,1}}, {{0,1}, {-1,0}}};

    // Transform basis functions into Voigt notation
    tensor<double,4> bf_voigt = outer(bf, voigt.transpose({2,0,1}));
    tensor<double,5> Dbf_voigt = outer(Dbf_g, voigt.transpose({2,0,1})).transpose({0,2,3,4,1});

    //-----------------------------------------------------------
    // [7] COMPUTE DERIVATIVE FIELDS
    //-----------------------------------------------------------

    // Compute thickness and its gradient
    double h = bf * nbor_h;
    double h0 = bf * nbor_h0;
    tensor<double,1> Dh  = nbor_h * Dbf_g;
    tensor<double,1> Dh0 = nbor_h0 * Dbf_g;

    // Compute nematic tensor Q and its spatial gradient
    tensor<double,2> q  = product(bf_voigt, nbor_q,  {{0,0},{1,1}});
    tensor<double,2> q0 = product(bf_voigt, nbor_q0, {{0,0},{1,1}});
    tensor<double,3> Dq = product(nbor_q, Dbf_voigt, {{0,0},{1,1}});
    tensor<double,3> Dq_0 = product(nbor_q0, Dbf_voigt, {{0,0},{1,1}});

    // Compute velocity and its gradient (rate-of-deformation tensor)
    tensor<double,1> v  = bf * nbor_v;
    tensor<double,2> Dv = product(nbor_v, Dbf_g, {{0,0}});
    double divv = trace(Dv); // Velocity divergence

    //-----------------------------------------------------------
    // [8] COMPUTE STRESS AND ROTATION TERMS
    //-----------------------------------------------------------

    // Compute rate-of-deformation tensor (symmetrized velocity gradient)
    tensor<double,2> rodt = 0.5 * (Dv + Dv.transpose({1,0}));
    tensor<double,4> dvrodt = 0.5 * (outer(Dbf_g, id2d).transpose({0,3,1,2}) + outer(Dbf_g, id2d).transpose({0,3,2,1}));

    // Compute vorticity tensor (anti-symmetric velocity gradient)
    tensor<double,2> omega = 0.5 * (Dv - Dv.transpose({1,0}));
    tensor<double,4> dvomega = 0.5 * (outer(Dbf_g, id2d).transpose({0,3,2,1}) - outer(Dbf_g, id2d).transpose({0,3,1,2}));

    //-----------------------------------------------------------
    // [9] COMPUTE TIME EVOLUTION TERMS
    //-----------------------------------------------------------

    // Compute Jaumann derivative of the Q tensor
    tensor<double,2> Jq = (q - q0) / deltat + Dq_0 * v - product(omega, q0, {{1,0}}) + product(q0, omega, {{1,0}});

    // Compute its derivatives with respect to Q and velocity
    tensor<double,4> dqJq = bf_voigt / deltat;
    tensor<double,4> dvJq = outer(bf, Dq_0.transpose({2,0,1})) - 2 * product(dvomega, q0, {{3,0}});

    // Compute polymerization-depolymerization rate adjustment based on alignment
    double S2 = 2.0 * product(q, q, {{0,0},{1,1}});
    double S2_0 = 2.0 * product(q0, q0, {{0,0},{1,1}});
    double S = sqrt(S2);
    double kd = computekd(kd0, sqrt(S2_0));

    // Compute updated thickness evolution
    double h_ = h0 + deltat * (- (v * Dh) - h * divv + kp - kd * h);

    //-----------------------------------------------------------
    // [10] FREE ENERGY TERMS (SUSCEPTIBILITY-BASED)
    //-----------------------------------------------------------

    // Compute susceptibility terms
    double sus2{}, dsus2{}, ddsus2{};
    double sus4{}, dsus4{}, ddsus4{};
    sus2 = sus20;
    sus4 = sus40;

    // Compute gradient and Hessian terms for free energy
    tensor<double,2> dvh_ = -deltat * (outer(bf, Dh) + h * Dbf_g);
    tensor<double,1> dhh_ = -deltat * (Dbf_g * v + bf * (divv + kd));
    tensor<double,3> dvdhh_ = -deltat * (outer(bf, Dbf_g.T()) + outer(Dbf_g, bf));

    tensor<double,2> qbfvoigt = product(q, bf_voigt, {{0,2},{1,3}});

    // Compute Rayleighian (dissipation function)
    rayleighian += jac * h * (sus2 * S2 + sus4 * S2 * S2);

    // Compute free energy for tracking
    freeEnergyPotential += jac * h * (sus2 * S2 + sus4 * S2 * S2);

    // Compute gradients for system matrix
    Bk(all, range(1,2)) += 2.0 * (jac * h0 * (sus2 + 2.0 * sus4 * S2)) * qbfvoigt;
    Bk(all, range(3,4)) += (0.5 * jac * S2 * ((sus2 + sus4 * S2) + h * (dsus2 + dsus4 * S2))) * dvh_;

    // Compute Hessian contributions for stability
    Ak(all, range(1,2), all, range(1,2)) += 2.0 * jac * h0 * ((sus2 + 2.0 * sus4 * S2) * 
        product(bf_voigt, bf_voigt, {{2,2},{3,3}}) + 8.0 * sus4 * outer(qbfvoigt, qbfvoigt));

    Ak(all, range(1,2), all, range(3,4)) += 2.0 * jac * ((sus2 + 2.0 * sus4 * S2) + h_ * (dsus2 + 2.0 * dsus4 * S2)) * outer(qbfvoigt, dvh_);
    Ak(all, range(3,4), all, range(1,2)) += 2.0 * jac * ((sus2 + 2.0 * sus4 * S2) + h * (dsus2 + 2.0 * dsus4 * S2)) * outer(dvh_, qbfvoigt);
    Ak(all, range(3,4), all, range(3,4)) += (0.5 * jac * S2 * (2.0 * (dsus2 + dsus4 * S2) + h_ * (ddsus2 + ddsus4 * S2))) * outer(dvh_, dvh_);

     /------------------------------------------------
        //[6] FREE ENERGY (FRANK)
    //-----------------------------------------------------------
    // [6] FREE ENERGY CONTRIBUTION (FRANK ENERGY)
    //-----------------------------------------------------------

    // Compute Q-tensor gradient magnitude
    double DqDq = product(Dq, Dq, {{0,0}, {1,1}, {2,2}});

    // Compute divergence of Q-tensor
    tensor<double,1> div_q = Dq(0, all, 0) + Dq(1, all, 1);
    double divqdivq = product(div_q, div_q, {{0,0}});

    // Compute gradient and Hessian terms for Frank elasticity energy
    tensor<double,2> dq_qDqDq = product(bf_voigt, product(Dq, Dq, {{0,0}, {1,1}}), {{2,0}, {3,1}});
    dq_qDqDq += 2 * product(q, product(Dbf_voigt, Dq, {{2,0}, {3,1}}), {{0,2}, {1,3}});

    tensor<double,4> dqdq_qDqDq = 2 * product(bf_voigt, product(Dbf_voigt, Dq, {{2,0}, {3,1}}), {{2,2}, {3,3}});
    dqdq_qDqDq += 2 * (product(q, product(Dbf_voigt, Dbf_voigt, {{2,2}, {3,3}}), {{0,2}, {1,5}}) +
                        product(bf_voigt, product(Dbf_voigt, Dq, {{2,0}, {3,1}}), {{2,2}, {3,3}}).transpose({2,3,0,1}));

    // Compute divergence terms
    tensor<double,3> dq_divq = Dbf_voigt(all, all, 0, all, 0) + Dbf_voigt(all, all, 1, all, 1);
    tensor<double,2> dq_divqdivq = product(dq_divq, div_q, {{2,0}});
    tensor<double,4> dqdq_divqdivq = product(dq_divq, dq_divq, {{2,2}});

    // Add Frank energy contributions to system matrices
    Bk(all, range(1,2)) += kappa_splay * (0.5 * jac * h0 * frank) * dq_divqdivq;
    Ak(all, range(1,2), all, range(1,2)) += kappa_splay * (0.5 * jac * h0 * frank) * dqdq_divqdivq;

    Bk(all, range(1,2)) += kappa * (0.5 * jac * h0 * frank) * dq_qDqDq;
    Ak(all, range(1,2), all, range(1,2)) += kappa * (0.5 * jac * h0 * frank) * dqdq_qDqDq;

    Bk(all, range(1,2)) -= kappa_splay * (0.5 * jac * h0 * frank) * dq_qDqDq;
    Ak(all, range(1,2), all, range(1,2)) -= kappa_splay * (0.5 * jac * h0 * frank) * dqdq_qDqDq;

    // Compute transformed Q-gradient for further terms
    tensor<double,2> DqDbfvoigt = product(Dq, Dbf_voigt, {{0,2}, {1,3}, {2,4}});

    //-----------------------------------------------------------
    // [6.1] ADDITION TO RAYLEIGHIAN (DISSIPATION FUNCTION)
    //-----------------------------------------------------------
    rayleighian += 0.5 * jac * h * frank * DqDq;
    freeEnergyPotential += 0.5 * jac * h * frank * DqDq;

    //-----------------------------------------------------------
    // [6.2] GRADIENT CONTRIBUTIONS
    //-----------------------------------------------------------
    Bk(all, range(1,2)) += jac * h0 * frank * (kappa_iso + 0 * (kappa + kappa_splay) * S2) * DqDbfvoigt;

    //-----------------------------------------------------------
    // [6.3] HESSIAN CONTRIBUTIONS
    //-----------------------------------------------------------
    Ak(all, range(1,2), all, range(1,2)) += jac * h0 * frank * (kappa_iso + 0 * (kappa + kappa_splay) * S) * 
                                            product(Dbf_voigt, Dbf_voigt, {{2,2}, {3,3}, {4,4}});

    //-----------------------------------------------------------
    // [8] POWER POTENTIAL (ANISOTROPIC TERM)
    //-----------------------------------------------------------

    // Contribution to Rayleighian and Power potential
    rayleighian += (jac * h0 * lambda_aniso) * product(q0, Dv, {{0,0}, {1,1}});
    power += (jac * h0 * lambda_aniso) * product(q0, Dv, {{0,0}, {1,1}});

    // Gradient contributions
    Bk(all, range(3,4)) += (jac * h0 * lambda_aniso) * (Dbf_g * q0);

    //-----------------------------------------------------------
    // [10] SHEAR DISSIPATION
    //-----------------------------------------------------------

    // Compute and add shear dissipation contributions
    rayleighian += (0.5 * jac * h0 * visc) * product(rodt, rodt, {{0,0}, {1,1}});
    dissipation += (0.5 * jac * h0 * visc) * product(rodt, rodt, {{0,0}, {1,1}});

    // Gradient contributions
    Bk(all, range(3,4)) += (jac * h0 * visc) * product(rodt, dvrodt, {{0,2}, {1,3}});

    // Hessian contributions
    Ak(all, range(3,4), all, range(3,4)) += (jac * h0 * visc) * product(dvrodt, dvrodt, {{2,2}, {3,3}});

    //-----------------------------------------------------------
    // [12] ROTATIONAL DISSIPATION
    //-----------------------------------------------------------

    // Contribution to Rayleighian and Dissipation tracking
    rayleighian += (0.5 * jac * h0 * rvisc) * product(Jq, Jq, {{0,0}, {1,1}});
    dissipation += (0.5 * jac * h0 * rvisc) * product(Jq, Jq, {{0,0}, {1,1}});

    // Gradient contributions
    Bk(all, range(1,2)) += (jac * h0 * rvisc) * product(Jq, dqJq, {{0,2}, {1,3}});
    Bk(all, range(3,4)) += (jac * h0 * rvisc) * product(Jq, dvJq, {{0,2}, {1,3}});

    // Hessian contributions
    Ak(all, range(1,2), all, range(1,2)) += (jac * h0 * rvisc) * product(dqJq, dqJq, {{2,2}, {3,3}});
    Ak(all, range(1,2), all, range(3,4)) += (jac * h0 * rvisc) * product(dqJq, dvJq, {{2,2}, {3,3}});
    Ak(all, range(3,4), all, range(1,2)) += (jac * h0 * rvisc) * product(dvJq, dqJq, {{2,2}, {3,3}});
    Ak(all, range(3,4), all, range(3,4)) += (jac * h0 * rvisc) * product(dvJq, dvJq, {{2,2}, {3,3}});

    //-----------------------------------------------------------
    // [13] COUPLED DISSIPATION
    //-----------------------------------------------------------

    // Contribution to Rayleighian and Dissipation tracking
    rayleighian += (jac * h0 * cvisc) * product(rodt, Jq, {{0,0}, {1,1}});
    dissipation += (jac * h0 * cvisc) * product(rodt, Jq, {{0,0}, {1,1}});

    // Gradient contributions
    Bk(all, range(1,2)) += (jac * h0 * cvisc) * product(rodt, dqJq, {{0,2}, {1,3}});
    Bk(all, range(3,4)) += (jac * h0 * cvisc) * product(Jq, dvrodt, {{0,2}, {1,3}});
    Bk(all, range(3,4)) += (jac * h0 * cvisc) * product(rodt, dvJq, {{0,2}, {1,3}});

    // Hessian contributions
    Ak(all, range(1,2), all, range(3,4)) += (jac * h0 * cvisc) * product(dqJq, dvrodt, {{2,2}, {3,3}});
    Ak(all, range(3,4), all, range(3,4)) += (jac * h0 * cvisc) * product(dvJq, dvrodt, {{2,2}, {3,3}});

    //-----------------------------------------------------------
    // [14] FRICTION
    //-----------------------------------------------------------

    // Contribution to Rayleighian and Dissipation tracking
    rayleighian += (0.5 * jac * h0 * fric) * v * v;
    dissipation += (0.5 * jac * h0 * fric) * v * v;

    // Gradient contributions
    Bk(all, range(3,4)) += (jac * h0 * fric) * outer(bf, v);

    // Hessian contributions
    Ak(all, range(3,4), all, range(3,4)) += (jac * h0 * fric) * outer(outer(bf, id2d), bf).transpose({0,1,3,2});


   double  pre = bf_p * nborDOFs_p;

   Bk(all,range(3,4)) += (jac * pre) * Dbf_g ;
   Bk1 += (jac * divv) * bf_p ;
   //[5.2]  HESSIAN
   Ak01(all,range(3,4),all) += jac * outer(Dbf_g,bf_p) ;
   Ak10(all,all,range(3,4)) += jac * outer(bf_p ,Dbf_g) ;


}

/**
 * @brief Sets velocity boundary conditions for a circular domain.
 * 
 * This function applies constraints on velocity components for nodes near a circular boundary.
 * It also computes and normalizes the nematic director field based on angular coordinates.
 * 
 * @param dofHand Reference-counted pointer to the DOFsHandler object handling degrees of freedom.
 * @param userStr Reference-counted pointer to the UserStructure containing model parameters.
 */
void setCircleVelBC(Teuchos::RCP<hiperlife::DOFsHandler> dofHand, 
                    Teuchos::RCP<hiperlife::UserStructure> userStr)
{
    using namespace hiperlife;

    /// Loop over all local nodes in the mesh
    for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
    {
        /// Retrieve geometric properties of the node
        int crease = dofHand->mesh->nodeCrease(i, IndexType::Local);  // Node crease index (boundary marker)
        double x = dofHand->mesh->nodeCoord(i, 'x', IndexType::Local); // X-coordinate of node
        double y = dofHand->mesh->nodeCoord(i, 'y', IndexType::Local); // Y-coordinate of node

        /// Set initial boundary scalar value
        double S0Boundary = 1.0;    

        /// Compute angle theta in polar coordinates
        double theeta = atan(y / x);

        /// Compute nematic director field components
        double n1 = -sin(theeta); 
        double n2 = 3 * cos(theeta); 

        /// Normalize the nematic director field
        double n_mag = sqrt(n1 * n1 + n2 * n2);  
        n1 = n1 / n_mag;  
        n2 = n2 / n_mag;  

        /// Compute Q-tensor components
        double Q1 = n1 * n1 - 0.5; 
        double Q2 = n1 * n2;  

        /// Set initial values for Q-tensor components
        dofHand->nodeDOFs->setValue(1, i, IndexType::Local, 0.0);  
        dofHand->nodeDOFs->setValue(2, i, IndexType::Local, 0.0);  

        /// Apply velocity constraints for nodes within a circular region (radius 0.05, centered at y=0.5)
        if ((y - 0.5) * (y - 0.5) + x * x < 0.05 * 0.05)
        {
            dofHand->setConstraint(1, i, IndexType::Local, 0.0);  // Constrain velocity component 1
            dofHand->setConstraint(2, i, IndexType::Local, 0.0);  // Constrain velocity component 2
        }
    }
}

/**
 * @brief Applies boundary conditions for the 2D cortex nematic model.
 * 
 * This function enforces constraints on velocity and nematic order at the boundary of the domain.
 * It computes geometric properties, normalizes the boundary conditions, and applies penalization terms
 * to control velocity and nematic order parameters at the boundary.
 * 
 * @param fillStr Reference-counted pointer to the FillStructure object containing mesh and problem data.
 */
void LS_CortexNematic_Border(Teuchos::RCP<hiperlife::FillStructure> fillStr)
{
    using namespace hiperlife;
    using namespace hiperlife::Tensor;

    //-----------------------------------------------------------
    // [1] INPUT DATA
    //-----------------------------------------------------------

    /// Extract sub-fill structure for the cortex nematic model
    auto& subFill = (*fillStr)["cortexHand"];
    int nDim = subFill.nDim;     // Number of spatial dimensions
    int pDim = subFill.pDim;     // Polynomial dimension
    int eNN  = subFill.eNN;      // Number of element nodes
    int numDOFs = subFill.numDOFs; // Number of degrees of freedom per element
    int numAuxF = subFill.numAuxF; // Number of auxiliary fields

    /// Model parameters from user-defined structure
    double uPoly = fillStr->userStr->dparam[18]; // Polynomial order factor
    double v_pen =  1E6;  // Velocity penalization coefficient
    double q_pen =  1.0;  // Nematic order parameter penalization coefficient
    double r_out = 1.0;   // Radius of the outer boundary

    /// Extract boundary tangents from mesh
    std::vector<double> tr = subFill.tangentsBoundaryRef();
    tensor<double,1> tangentRef(tr.data(), 2);                  

    /// Initialize tensor structures for neighbor elements
    tensor<double,2> nborCoords(subFill.nborCoords.data(), eNN, nDim);
    tensor<double,2> nborDOFs0(subFill.nborDOFs0.data(), eNN, numDOFs);
    tensor<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);
    tensor<double,2> nborAuxF(subFill.nborAuxF.data(), eNN, numAuxF);

    /// Basis function values and derivatives
    tensor<double,1> bf(subFill.getDer(0), eNN);
    tensor<double,2> Dbf_l(subFill.getDer(1), eNN, pDim);
    tensor<double,3> DDbf_l(subFill.getDer(2), eNN, pDim, pDim);

    /// Transformation matrices for spatial derivatives
    tensor<double,3> DDbf_g(eNN, pDim, pDim);
    tensor<double,1> Lapbf_g(eNN);

    /// Define local stiffness and load contribution matrices
    tensor<double,2> Bk(fillStr->Bk(0).data(), eNN, numDOFs);
    tensor<double,4> Ak(fillStr->Ak(0,0).data(), eNN, numDOFs, eNN, numDOFs);

    //-----------------------------------------------------------
    // [2] GEOMETRY PROCESSING
    //-----------------------------------------------------------

    /// Compute element center coordinates
    tensor<double,1> x = bf * nborCoords;

    /// Compute transformation matrix (from reference to spatial coordinates)
    tensor<double,2> T = nborCoords(all, range(0,1)).T() * Dbf_l;
    tensor<double,1> tangent = T * tangentRef;
    double normt = sqrt(tangent(0)*tangent(0) + tangent(1)*tangent(1));

    /// Compute boundary normal and unit tangent vectors
    tensor<double,1> bnormal = {tangent(1), -tangent(0)};
    bnormal /= normt;
    tensor<double,1> unit_tangent = {bnormal(1), -bnormal(0)};

    /// Compute global derivatives of basis functions               
    double jac;
    tensor<double,2> Dbf_g = Dbf_l * T.inv();

    //-----------------------------------------------------------
    // [3] VARIABLE INITIALIZATION
    //-----------------------------------------------------------

    /// [3.1] Auxiliary Variables
    tensor<double,1> nbor_h  = nborDOFs(all, 0);
    tensor<double,1> nbor_h0 = nborDOFs0(all, 0);
    tensor<double,2> nbor_q  = nborDOFs(all, range(1,2));
    tensor<double,2> nbor_q0 = nborDOFs0(all, range(1,2));
    tensor<double,2> nbor_v = nborDOFs(all, range(3,4));

    /// Define identity tensor and Voigt notation tensors
    tensor<double,2> id2d = {{1.0, 0.0}, {0.0, 1.0}};
    tensor<double,3> voigt = {{{1, 0}, {0, 1}}, {{0, 1}, {-1, 0}}};
    tensor<double,4> bf_voigt = outer(bf, voigt.transpose({2,0,1}));
    tensor<double,5> Dbf_voigt = outer(Dbf_g, voigt.transpose({2,0,1})).transpose({0,2,3,4,1});

    /// Compute nematic order parameters at nodes
    tensor<double,2> q  = product(bf_voigt, nbor_q, {{0,0},{1,1}});
    tensor<double,2> q0 = product(bf_voigt, nbor_q0, {{0,0},{1,1}});
    tensor<double,3> Dq = product(nbor_q, Dbf_voigt, {{0,0},{1,1}});
    tensor<double,3> Dq_0 = product(nbor_q0, Dbf_voigt, {{0,0},{1,1}});

    /// [3.2] State/Process Variables
    double h0 = bf * nbor_h0;  // Thickness field
    tensor<double,1> v = bf * nbor_v;  // Velocity field
    double vn = v * bnormal;  // Normal velocity
    double vp = uPoly;  // Predefined velocity magnitude
    double rad = sqrt(x(0) * x(0) + x(1) * x(1));  // Radial coordinate

    /// Define nematic order at boundary
    double S0Boundary = 1.0;
    double theeta = M_PI / 2 + atan(x(1) / x(0));
    double n1 = cos(theeta);
    double n2 = sin(theeta);
    double Q1 = (S0Boundary * (n1 * n1 - n2 * n2)) / (n1 * n1 + n2 * n2);
    double Q2 = (2 * S0Boundary * n1 * n2) / (n1 * n1 + n2 * n2);
    tensor<double,2> q_f = {{Q1, Q2}, {Q2, -Q1}};
    tensor<double,2> qbfvoigt = product(q - q_f, bf_voigt, {{0,2}, {1,3}});

    //-----------------------------------------------------------
    // [4] APPLY BOUNDARY CONDITIONS
    //-----------------------------------------------------------

    {
        double rayleighian = 0.0;

        /// Apply velocity penalization at the boundary
        rayleighian += normt * h0 * 0.5 * v_pen * (vn * vn);

        /// Compute gradient contribution to system matrix
        Bk(all, range(3,4)) += normt * h0 * v_pen * vn * outer(bf, bnormal);

        /// Compute Hessian contribution to system matrix
        Ak(all, range(3,4), all, range(3,4)) += normt * h0 * v_pen * outer(outer(bf, bnormal), outer(bf, bnormal));

        /// Apply nematic order penalization at the boundary
        Bk(all, range(1,2)) += v_pen * normt * h0 * 
            product((q - outer(unit_tangent, unit_tangent) + 0.5 * Identity(2)), bf_voigt, {{0,2}, {1,3}});

        /// Compute Hessian contribution for nematic order constraints
        Ak(all, range(1,2), all, range(1,2)) += v_pen * normt * h0 * product(bf_voigt, bf_voigt, {{2,2}, {3,3}});
    }                                                                                                                                                             
}
