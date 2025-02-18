#ifndef Aux_src
#define Aux_src  // Include guard to prevent multiple inclusions of this header file

/// Include necessary libraries and headers
#include <Teuchos_RCP.hpp>                  // Reference-counted pointers from the Teuchos package
#include <Teuchos_CommandLineProcessor.hpp> // Command-line processor from the Teuchos package

#include "hl_FillStructure.h"   // Data structures for filling computational grids
#include "hl_HiPerProblem.h"    // HiPerProblem class for problem formulation
#include "hl_Tensor.h"          // Tensor operations

#include "hl_DOFsHandler.h"     // Handles degrees of freedom (DOFs) for numerical solutions
#include "hl_UserStructure.h"   // Structure for storing user-defined parameters

/// Function declarations for defining the computational model

/**
 * @brief Defines the element filling structure for the 2D cortex nematic model.
 * @param fillStr Reference-counted pointer to the FillStructure object.
 */
void LS_CortexNematic_2D(Teuchos::RCP<hiperlife::FillStructure> fillStr);

/**
 * @brief Applies velocity boundary conditions for a circular domain.
 * @param dofHand Reference-counted pointer to the DOFsHandler object.
 * @param userStr Reference-counted pointer to the UserStructure containing model parameters.
 */
void setCircleVelBC(Teuchos::RCP<hiperlife::DOFsHandler> dofHand, 
                    Teuchos::RCP<hiperlife::UserStructure> userStr);

/**
 * @brief Applies nematic boundary conditions for a circular domain.
 * @param dofHand Reference-counted pointer to the DOFsHandler object.
 * @param userStr Reference-counted pointer to the UserStructure containing model parameters.
 */
void setCircleNematicBC(Teuchos::RCP<hiperlife::DOFsHandler> dofHand, 
                        Teuchos::RCP<hiperlife::UserStructure> userStr);

/**
 * @brief Applies thickness boundary conditions for a circular domain.
 * @param dofHand Reference-counted pointer to the DOFsHandler object.
 * @param userStr Reference-counted pointer to the UserStructure containing model parameters.
 */
void setCircleThiBC(Teuchos::RCP<hiperlife::DOFsHandler> dofHand, 
                    Teuchos::RCP<hiperlife::UserStructure> userStr);

/**
 * @brief Defines the element filling structure for the border of the 2D cortex nematic model.
 * @param fillStr Reference-counted pointer to the FillStructure object.
 */
void LS_CortexNematic_Border(Teuchos::RCP<hiperlife::FillStructure> fillStr);

/**
 * @brief Post-processing function for the simulation.
 * @param fillStr Reference-counted pointer to the FillStructure object.
 */
void LS_post(Teuchos::RCP<hiperlife::FillStructure> fillStr);

#endif // Aux_src
