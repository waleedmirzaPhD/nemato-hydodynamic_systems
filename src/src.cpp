
/// C++ standard library headers
#include <iostream>   // Input-output stream library
#include <mpi.h>      // MPI (Message Passing Interface) library for parallel computing
#include <fstream>    // File stream operations

/// Trilinos library headers (for numerical computations and parallel solvers)
#include <Teuchos_RCP.hpp>  // Reference-counted pointers for memory management

/// HiPerLife library headers (presumably for high-performance numerical simulations)
#include "hl_TypeDefs.h"                     // Type definitions used across the project
#include "hl_Geometry.h"                     // Geometric definitions and operations
#include "hl_StructMeshGenerator.h"          // Structured mesh generation utilities
#include "hl_DistributedMesh.h"              // Mesh handling for distributed computing
#include "hl_FillStructure.h"                // Data structures for filling computational grids
#include "hl_DOFsHandler.h"                  // Degrees of freedom (DOF) handler for FEM computations
#include "hl_HiPerProblem.h"                 // Base class for defining HiPerLife problems
#include "hl_LinearSolver_Direct_Amesos.h"   // Direct linear solver using Amesos library
#include "hl_Tensor.h"                       // Tensor operations for computational mechanics
#include "hl_MeshLoader.h"                   // Utility for loading meshes from files
#include <Teuchos_RCP.hpp>                   // Reference-counted pointers (repeated inclusion)
#include <Teuchos_CommandLineProcessor.hpp>  // Command-line argument parsing utilities
#include "hl_ConfigFile.h"                   // Configuration file handling
#include "hl_ConsistencyCheck.h"             // Utilities for checking consistency in simulations
#include "hl_ConsistencyCheck.h"             // Duplicate inclusion of the same header
#include "Aux_src.h"              // Auxiliary computations for nematic models in 2D
#include "hl_LinearSolver_Direct_Amesos.h"   // Duplicate inclusion of Amesos direct solver
#include "hl_LinearSolver_Iterative_AztecOO.h"  // Iterative linear solver using AztecOO
#include "hl_NonlinearSolver_NewtonRaphson.h"   // Newton-Raphson solver for nonlinear problems
#include "hl_Remesher.h"                        // Mesh refinement and remeshing utilities
#include "hl_UnstructVtkMeshGenerator.h"        // Unstructured VTK mesh generator
#include "hl_LinearSolver_Direct_MUMPS.h"       // Direct linear solver using MUMPS library




int main ( int argc, char *argv[]  )
{
    /// Using directives
    using namespace std;          // Standard C++ namespace
    using namespace hiperlife;    // HiPerLife namespace for the specific simulation framework
    using Teuchos::rcp;          // Reference-counted pointer from Teuchos (Trilinos)
    using Teuchos::RCP;          // Smart pointer type from Teuchos

    /// Initialize MPI (Message Passing Interface for parallel computing)
    MPI_Init(nullptr, nullptr);

    /// Boolean flags for output control
    bool printVtk{true};         // Flag to determine if VTK output should be printed
    bool printFile{true};        // Flag to determine if file output should be printed
    bool diagnosticFlag{false};  // Flag for diagnostic mode

    /// Read configuration file
    const char *config_filename = argv[1]; // Get configuration file from command-line argument
    ConfigFile config(config_filename);    // Load configuration file

    /// Read restart parameter from config file
    int restart{};
    config.readInto(restart, "restart");

    /// Define file path for storing CSV output
    string csvPathDiss = "globalIntegralsDiss.csv";
    config.readInto(csvPathDiss, "csvPathDiss");

    /// Geometric parameters for structured mesh (not defined explicitly)

    /// Model parameters (physical properties for the simulation)
    double visc = 20.0;   // Viscosity parameter
    config.readInto(visc, "visc");

    double cvisc = 3.0;   // Secondary viscosity parameter
    config.readInto(cvisc, "cvisc");

    double rvisc = 1.0;   // Rotational viscosity
    config.readInto(rvisc, "rvisc");

    double frank = 1.0;   // Frank constant for elasticity
    config.readInto(frank, "frank");

    /// Lambda parameters for isotropic, anisotropic, and rotational deformations
    double lambda_iso = 0.0;
    config.readInto(lambda_iso, "lambda_iso");

    double lambda_aniso = 0.0;
    config.readInto(lambda_aniso, "lambda_aniso");

    double lambda_rot = 0.0;
    config.readInto(lambda_rot, "lambda_rot");

    /// Additional model parameters
    double kp = 0.2 * 0.1;  // Constant parameter
    config.readInto(kp, "kp");

    double kd = 0.1;  // Another model parameter
    config.readInto(kd, "kd");

    double fric = 0.0;  // Friction parameter
    config.readInto(fric, "fric");

    double sus20 = 200.0;  // Parameter related to the model's susceptibility
    config.readInto(sus20, "sus20");

    double sus40 = 800.0;  // Another susceptibility parameter
    config.readInto(sus40, "sus40");

    double hcrit = 0.25;  // Critical value for an unspecified property
    config.readInto(hcrit, "hcrit");

    /// Time integration parameters
    double deltat = 1E-2;  // Time step size
    config.readInto(deltat, "deltat");

    /// Numerical solver parameters
    string linTol = "1e-12";  // Linear solver tolerance
    config.readInto(linTol, "linTol");

    double resTol = 1e-5;  // Residual tolerance for convergence
    config.readInto(resTol, "resTol");

    double solTol = 1e-5;  // Solution tolerance
    config.readInto(solTol, "solTol");

    int maxIter = 10;  // Maximum number of iterations
    config.readInto(maxIter, "maxIter");

    string linImax = "10000";  // Maximum number of linear solver iterations
    config.readInto(linImax, "linImax");

    double maxDelt = 0.1;  // Maximum time step
    config.readInto(maxDelt, "maxDelt");

    double adaptiveStepTime = 0.975;  // Adaptive step size parameter
    config.readInto(adaptiveStepTime, "adaptiveStepTime");

    /// Output control parameters
    int nPrint = 1;  // Printing frequency for output
    config.readInto(nPrint, "nPrint");

    double stab = 1.0;  // Stabilization parameter
    config.readInto(stab, "stab");

    double heqb = 1;  // Equilibrium height
    config.readInto(heqb, "heqb");

    /// Consistency check flag
    bool cCheck{false};
    config.readInto(cCheck, "cCheck");

    /// Input file for mesh definition
    string fMesh{};
    config.readInto(fMesh, "fMesh");

    /// Output file names
    string solname_v;  // Solution file name (not initialized)
    string oname = "stressFiberCircle";  // Default output name
    string pname = "post";  // Default post-processing name
    config.readInto(oname, "oname");

    /// Time step and initial time
    int timeStep = 0;
    double time = 0.0;

    /// Mesh and geometry parameters
    int bfOrder = 1;  // Order of basis functions
    bool balanceMesh = true;  // Flag to determine mesh balancing

    /// Adjusted time step and max delta (commented-out alternative formulas)
    maxDelt = 1;  // (1./25.) * std::min((0.5*visc/lambda_aniso), rvisc/frank);
    cout << maxDelt << endl;  // Print maximum delta value

    /// Define user structure for model parameters
    RCP<UserStructure> userStr = rcp(new UserStructure);
    RCP<UserStructure> userNum = rcp(new UserStructure);

    /// Populate user structure with model parameters
    {
        userStr->dparam.resize(25);

        userStr->dparam[0] = deltat;
        userStr->dparam[1] = sus20;
        userStr->dparam[2] = sus40;
        userStr->dparam[3] = hcrit;

        userStr->dparam[4] = kp;
        userStr->dparam[5] = kd;
        userStr->dparam[6] = frank;

        userStr->dparam[7] = lambda_iso;
        userStr->dparam[8] = lambda_aniso;
        userStr->dparam[9] = lambda_rot;

        userStr->dparam[10] = visc;
        userStr->dparam[11] = rvisc;
        userStr->dparam[12] = cvisc;

        userStr->dparam[13] = fric;
        userStr->dparam[14] = stab;
        userStr->dparam[15] = phi;  // Uninitialized parameter
        userStr->dparam[16] = r_in;  // Uninitialized parameter
        userStr->dparam[17] = r_out;  // Uninitialized parameter
        userStr->dparam[18] = uPoly;  // Uninitialized parameter
        userStr->dparam[19] = t_stall;  // Uninitialized parameter
        userStr->dparam[20] = kosm;  // Uninitialized parameter
        userStr->dparam[22] = heqb;
        userStr->dparam[23] = L;  // Uninitialized parameter

        /// Numerical solver parameters
        userNum->dparam.resize(3);
        userNum->iparam.resize(3);
        userNum->dparam[0] = solTol;
        userNum->dparam[1] = resTol;
        userNum->iparam[0] = maxIter;
        userNum->iparam[1] = testCase;  // Uninitialized parameter
    }

    /// Create a new mesh loader instance
    RCP<MeshLoader> loadedMesh = rcp(new MeshLoader);

    /// Define mesh properties
    loadedMesh->setElemType(ElemType::Triang);  // Set element type to triangular
    loadedMesh->setBasisFuncType(BasisFuncType::Lagrangian);  // Set basis function type to Lagrangian
    loadedMesh->setBasisFuncOrder(bfOrder);  // Set order of basis functions
    loadedMesh->loadLegacyVtk(fMesh);  // Load mesh from a legacy VTK file

    /// Transform mesh coordinates using a lambda function
    loadedMesh->transformFree([](double x, double y) 
    {
        double xx, yy;
        xx = x; 
        yy = y;
        x =  5. * x;  // Scale x-coordinate by 5
        return std::make_tuple(x, y);
    });                                                         

    /// Create distributed meshes for parallel computing
    RCP<DistributedMesh> disMesh = rcp(new DistributedMesh);
    RCP<DistributedMesh> disMeshII = rcp(new DistributedMesh);
    RCP<DistributedMesh> disMeshPress;  // Mesh for pressure calculations

    /// Try to initialize and update the pressure mesh
    try
    {
        disMeshPress = rcp(new DistributedMesh);
        disMeshPress->setMesh(loadedMesh);  // Set the base mesh
        disMeshPress->setBalanceMesh(balanceMesh);  // Enable load balancing
        disMeshPress->Update();  // Update mesh structure
        disMeshPress->printFileLegacyVtk("DensityDependCortexNemacMesh_press");  // Output mesh

        if (disMeshPress->myRank() == 0)
            cout << "Dismesh successfully created." << endl;                        
    }
    catch (runtime_error)
    {
        throw runtime_error("Error in dismesh.");
        MPI_Finalize();
        return 1;
    }                                                                            

    /// Configure mesh refinement and balancing
    disMesh->setMeshRelation(MeshRelation::hRefin, disMeshPress);
    disMesh->setHRefinement(1);  // Set h-refinement level
    disMesh->setBalanceMesh(balanceMesh);  // Enable load balancing
    disMesh->Update();  // Apply updates to the distributed mesh

    /// Create degrees of freedom (DOFs) handlers
    RCP<DOFsHandler> dofHand;
    RCP<DOFsHandler> dhandP = rcp(new DOFsHandler(disMeshPress));  // Handler for pressure mesh

    /// Check if the simulation is starting from scratch
    if (restart == 0)
    {
        try
        {
            /// Initialize DOFs handler for the primary mesh
            dofHand = rcp(new DOFsHandler(disMesh));
            dofHand->setNameTag("cortexHand");  // Assign a name
            dofHand->setDOFs({"h", "q1", "q2", "vx", "vy"});  // Define DOFs
            dofHand->setNodeAuxF({"x","y"});  // Define auxiliary node fields
            dofHand->Update();  // Update DOFs handler

            /// Configure DOFs handler for pressure
            dhandP->setNameTag("dhandP");  // Assign a name
            dhandP->setDOFs({"p"});  // Define DOF for pressure
            dhandP->Update();  // Update handler

            if (dofHand->myRank() == 0)
                cout << "DOFsHandler successfully created." << endl;

            /// Apply boundary conditions
            for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
            {
                double x = dofHand->mesh->nodeCoord(i, 0, IndexType::Local);
                double y = dofHand->mesh->nodeCoord(i, 1, IndexType::Local);
                int nElem = dofHand->mesh->_adjcyNE->getNumNbors(i, IndexType::Local);
                
                // Mark nodes based on radial distance and number of adjacent elements
                if (sqrt(x*x + y*y) > 0.99 && (nElem == 3 || nElem == 4))
                {
                    dofHand->mesh->_nodeFlags->setValue(0, i, IndexType::Local, 1);
                }
                else
                {
                    dofHand->mesh->_nodeFlags->setValue(0, i, IndexType::Local, 0);
                }
            }   

            /// Update ghost values across distributed processes
            dofHand->mesh->_nodeFlags->UpdateGhosts();
            dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);
            dofHand->UpdateGhosts();                                                                 

        }
        catch (runtime_error err)
        {
            cout << dofHand->myRank() << ": DOFsHandler could not be created. " << err.what() << endl;
            MPI_Finalize();
            return 1;
        }

        /// Apply boundary conditions to the simulation
        setCircleVelBC(dofHand, userStr);
        setCircleThiBC(dofHand, userStr);

        /// Apply constraints to pressure DOFs
        if (disMeshPress->myRank() == 0)
            dhandP->setConstraint(0, 0, IndexType::Local, 0.0);

        /// Synchronize ghost DOFs across processes
        dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);
        dhandP->nodeDOFs0->setValue(dhandP->nodeDOFs);
        dhandP->UpdateGhosts();
    }

    /// Final ghost update and mesh output
    dofHand->mesh->_nodeFlags->UpdateGhosts();
    dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);
    dofHand->UpdateGhosts();
    dofHand->printFileLegacyVtk("InitialMesh_1", true);                                                             

    /// Initialize the HiPerLife problem
    RCP<HiPerProblem> hiperProbl = rcp(new HiPerProblem);
    hiperProbl->setUserStruct(userStr);
    hiperProbl->setDOFsHandlers({dofHand, dhandP});
    hiperProbl->setIntegration("Integ", {"cortexHand", "dhandP"});
    hiperProbl->setIntegration("BorderInteg", {"cortexHand"});
    hiperProbl->setCubatureGauss("Integ", 3);
    hiperProbl->setCubatureBorderGauss("BorderInteg", 2);

    /// Define element fillings based on consistency check
    if (cCheck)
    {
        hiperProbl->setElemFillings("Integ", ConsistencyCheck<LS_CortexNematic_2D>);
        hiperProbl->setElemFillings("BorderInteg", ConsistencyCheck<LS_CortexNematic_Border>);
    }
    else
    {
        hiperProbl->setElemFillings("Integ", LS_CortexNematic_2D);
        hiperProbl->setElemFillings("BorderInteg", LS_CortexNematic_Border);
    }

    /// Final update for the problem
    hiperProbl->Update();

    if (hiperProbl->myRank() == 0)
        cout << "HiperProblem successfully updated." << endl;

    /// Output initial condition
    string solName = oname + ".0";
    if (printVtk)
        dofHand->printFileLegacyVtk(solName, true);
    if (printFile)
        dofHand->printFile(solName, OutputMode::Text, true, 0.0);

    /// Set up MUMPS direct linear solver
    RCP<MUMPSDirectLinearSolver> linsolver = rcp(new MUMPSDirectLinearSolver());
    linsolver->setHiPerProblem(hiperProbl);
    linsolver->setMatrixType(MUMPSDirectLinearSolver::MatrixType::General);
    linsolver->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel);
    linsolver->setOrderingLibrary(MUMPSDirectLinearSolver::OrderingLibrary::Auto);
    linsolver->setVerbosity(MUMPSDirectLinearSolver::Verbosity::None);
    linsolver->setDefaultParameters();
    linsolver->setWorkSpaceMemoryIncrease(500);
    linsolver->Update();

    /// Set up Newton-Raphson nonlinear solver
    RCP<NewtonRaphsonNonlinearSolver> nonlinSolver = rcp(new NewtonRaphsonNonlinearSolver());
    nonlinSolver->setLinearSolver(linsolver);
    nonlinSolver->setMaxNumIterations(maxIter);
    nonlinSolver->setResTolerance(resTol);
    nonlinSolver->setSolTolerance(solTol);
    nonlinSolver->setLineSearch(false);
    nonlinSolver->setPrintIntermInfo(true);
    nonlinSolver->setConvRelTolerance(false);
    nonlinSolver->Update();
 
    /// Main time-stepping loop
    while (timeStep < totalTimeSteps)
    {
        /// Print time step information
        if (hiperProbl->myRank() == 0)
            cout << "    Time step " + to_string(timeStep) + " "
                << "deltat= " << userStr->dparam[0] 
                << " and maximum iteration " << maxIter << endl;

        /// Set initial guess for Newton-Raphson solver
        dofHand->nodeDOFs->setValue(dofHand->nodeDOFs0);
        hiperProbl->UpdateGhosts();

        /// Solve the nonlinear system using Newton-Raphson
        bool converged = nonlinSolver->solve();
        int nIter = userNum->iparam[1];  // Retrieve the number of iterations used

        /// Check if the solver converged
        if (converged)
        {
            /// Save the solution from this time step
            dofHand->nodeDOFs0->setValue(dofHand->nodeDOFs);
            dhandP->nodeDOFs0->setValue(dhandP->nodeDOFs);

            /// Transfer velocity values to auxiliary field for further processing
            for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
            {
                double vx = dofHand->nodeDOFs->getValue("vx", i, IndexType::Local);
                double vy = dofHand->nodeDOFs->getValue("vy", i, IndexType::Local);
                handII->nodeAuxF->setValue(0, i, IndexType::Local, vx);
                handII->nodeAuxF->setValue(1, i, IndexType::Local, vy);
            }

            /// Solve the secondary linear system
            linsolverII->solve();
            handII->nodeDOFs->setValue(problII->linProbl->GetLHS());
            handII->UpdateGhosts();                                      

            /// Adjust the time step size adaptively
            if (nIter <= maxIter)
                deltat /= adaptiveStepTime;  // Increase step size if convergence is fast
            else if (nIter > maxIter)
                deltat *= adaptiveStepTime;  // Decrease step size if convergence is slow
            
            /// Ensure time step does not exceed the maximum allowed
            if (deltat > maxDelt)
                deltat = maxDelt;

            /// Update simulation time variables
            timeStep++;
            time += deltat;

            /// Save results at specified intervals
            if (timeStep % nPrint == 0)
            { 
                solname_v = oname + "." + to_string(timeStep);
                string solname_post = pname + "." + to_string(timeStep);  
                
                /// Output VTK files if enabled
                if (printVtk)
                {
                    dofHand->printFileLegacyVtk(solname_v, true);
                    dhandP->printFileLegacyVtk(solname_post, true);
                }

                /// Output text files if enabled
                if (printFile)
                {
                    solname_v = oname + "." + to_string(timeStep);
                    dofHand->printFile(solname_v, OutputMode::Text, true, time);
                    string solname_p = oname + "P." + to_string(timeStep);
                    dhandP->printFile(solname_p, OutputMode::Text, true, time);
                }
            }
        }
        else
        {
            /// Reduce the time step size if the solution did not converge
            deltat *= adaptiveStepTime;
        }

        /// Adjust parameters at a specific time step (currently disabled)
        if (false) // timeStep == 70)
        {
            userStr->dparam[11] = 1000;  // Modify viscosity parameter
            userStr->dparam[8] = 0.1;    // Modify lambda_aniso
            maxDelt = 0.025;  // Adjust maximum allowable time step
        }

        /// Update stored time step size
        userStr->dparam[0] = deltat;
    }

    /// Save final time step results
    solname_v = oname + "." + to_string(timeStep + 1);
    if (printVtk)
        dofHand->printFileLegacyVtk(solname_v, true);
    if (printFile)
        dofHand->printFile(solname_v, OutputMode::Text, true, time + deltat);

    /// Finalize MPI communication and exit
    MPI_Finalize();
    return 0;

}
