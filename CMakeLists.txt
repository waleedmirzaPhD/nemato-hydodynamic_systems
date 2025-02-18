#------------------------------------------------------------
# CMakeLists for CortexNemac
#------------------------------------------------------------
file(GLOB SOURCE_CPP *.cpp)



get_filename_component(exName ${CMAKE_CURRENT_LIST_DIR} NAME)


set(exName "hl${exName}")

message(STATUS "exName:  ${exName} ")
#message(STATUS "link:    ${LIN} ")
#message(STATUS "Trilinos libs:    ${Trilinos_LIBRARIES} ")
#message(STATUS "HPLFE libs:    ${HPLFE_LIBRARIES} ")
#
add_executable(${exName} ${SOURCE_CPP})
target_link_libraries(${exName} ${Trilinos_LIBRARIES})
target_link_libraries(${exName} ${HPLFE_LIBRARIES})
#target_link_libraries(${programname} ${VTK_LIBRARIES})
#target_link_libraries(${programname} ${VTK_LIBRARIES})
target_link_libraries(${exName} ${VTK_LIBRARIES})
find_library(GMSH_LIB NAMES gmsh HINTS ${GMSH_BASE_PATH}/lib)
    if(GMSH_LIB)
        include_directories(SYSTEM ${GMSH_BASE_PATH}/include)
    target_link_libraries(${exName} ${GMSH_LIB})
    endif()

#
#
#
set_target_properties( ${exName} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${APPS_INSTALL_DIR})
install(TARGETS ${exName} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
