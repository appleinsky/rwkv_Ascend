
function(get_system_info SYSTEM_INFO)
if (UNIX)
  execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
  string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
  set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
elseif (WIN32)
  message(STATUS "System is Windows. Only for pre-build.")
else ()
  message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
endif ()
endfunction()

function(opbuild)
message(STATUS "Opbuild generating sources")
cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX" "OPS_SRC" ${ARGN})
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                -I ${ASCEND_CANN_PACKAGE_PATH}/include -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                RESULT_VARIABLE EXEC_RESULT
                OUTPUT_VARIABLE EXEC_INFO
                ERROR_VARIABLE  EXEC_ERROR
)
if (${EXEC_RESULT}) 
  message("build ops lib info: ${EXEC_INFO}")
  message("build ops lib error: ${EXEC_ERROR}")
  message(FATAL_ERROR "opbuild run failed!")
endif()
set(proj_env "")
set(prefix_env "")
if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
  set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
endif()
if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
  set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
endif()
execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                        ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                RESULT_VARIABLE EXEC_RESULT
                OUTPUT_VARIABLE EXEC_INFO
                ERROR_VARIABLE  EXEC_ERROR
)
if (${EXEC_RESULT}) 
  message("opbuild ops info: ${EXEC_INFO}")
  message("opbuild ops error: ${EXEC_ERROR}")
endif()
message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
add_custom_command(OUTPUT ${OPINFO_OUTPUT}
    COMMAND mkdir -p ${opinfo_file_path}
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/parse_ini_to_json.py
            ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
)
add_custom_target(${OPINFO_TARGET} ALL
    DEPENDS ${OPINFO_OUTPUT}
)
install(FILES ${OPINFO_OUTPUT}
        DESTINATION ${OPINFO_INSTALL_DIR}
)
endfunction()

function(add_ops_impl_target)
cmake_parse_arguments(OPIMPL "" "TARGET;OPS_INFO;IMPL_DIR;OUT_DIR;INSTALL_DIR" "OPS_BATCH;OPS_ITERATE" ${ARGN})
set(AUTO_SYNC "auto_sync_true") 
add_custom_command(OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND mkdir -p ${OPIMPL_OUT_DIR}/dynamic
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
            ${OPIMPL_OPS_INFO}
            \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\"
            ${OPIMPL_IMPL_DIR}
            ${OPIMPL_OUT_DIR}/dynamic
            ${AUTO_SYNC}
    COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
    COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp
    DEPENDS ${OPIMPL_OPS_INFO}
            ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
)
add_custom_target(${OPIMPL_TARGET} ALL
    DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp)
if (${ENABLE_SOURCE_PACKAGE})
  install(DIRECTORY ${OPIMPL_OUT_DIR}/dynamic
      DESTINATION ${OPIMPL_INSTALL_DIR}
  )
endif()
endfunction()

function(add_ops_replay_targets)
cmake_parse_arguments(OPREPLAY "" "OPS_INFO;COMPUTE_UNIT;IMPL_DIR;OUT_DIR;INSTALL_DIR" "OPS_BATCH;OPS_ITERATE" ${ARGN})
# ccec compile options
set(ccec_base_opts -c -O2 --cce-aicore-only -mllvm -cce-aicore-function-stack-size=16000
                   -mllvm -cce-aicore-record-overflow=false -std=c++17)
set(ccec_extopts_ascend310p --cce-aicore-arch=dav-m200 -mllvm -cce-aicore-fp-ceiling=2)
set(ccec_extopts_ascend910 --cce-aicore-arch=dav-c100)
set(ccec_extopts_ascend910b --cce-aicore-arch=dav-c220-cube)
file(MAKE_DIRECTORY ${OPREPLAY_OUT_DIR})
execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_replay_build.py
                        ${OPREPLAY_OPS_INFO}
                        "${OPREPLAY_OPS_BATCH}" "${OPREPLAY_OPS_ITERATE}"
                        ${OPREPLAY_IMPL_DIR}
                        ${OPREPLAY_OUT_DIR}
                        ${OPREPLAY_COMPUTE_UNIT}
)
file(GLOB replay_kernel_entries ${OPREPLAY_OUT_DIR}/*.cce)
if (NOT "${replay_kernel_entries}x" STREQUAL "x")
  foreach(replay_kernel_file ${replay_kernel_entries})
    get_filename_component(replay_kernel_file_name "${replay_kernel_file}" NAME)
    string(REPLACE "_entry.cce" "" op_kerne_name ${replay_kernel_file_name})
    file(GLOB replay_lib_src ${OPREPLAY_OUT_DIR}/${op_kerne_name}*.cpp)
    set(OP_TILING_DATA_H_PATH ${OPREPLAY_OUT_DIR}/${op_kerne_name}_tiling_data.h)
    add_library(replay_${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT} SHARED ${replay_lib_src})
    if(EXISTS ${OP_TILING_DATA_H_PATH})
      target_compile_options(replay_${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT} PRIVATE
        -include ${OP_TILING_DATA_H_PATH}
      )
    endif()
    target_compile_definitions(replay_${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT} PRIVATE
      ${op_kerne_name}=${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT}
    )
    target_compile_options(replay_${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT} PRIVATE
      -D__ASCENDC_REPLAY__
    )
    target_link_libraries(replay_${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT} PRIVATE intf_pub
      tikreplaylib::${OPREPLAY_COMPUTE_UNIT}
      register
    )
    add_custom_command(OUTPUT ${OPREPLAY_OUT_DIR}/${op_kerne_name}_entry_${OPREPLAY_COMPUTE_UNIT}.o
                       COMMAND ccec ${ccec_base_opts} ${ccec_extopts_${OPREPLAY_COMPUTE_UNIT}} ${replay_kernel_file}
                               -o ${OPREPLAY_OUT_DIR}/${op_kerne_name}_entry_${OPREPLAY_COMPUTE_UNIT}.o
                       DEPENDS ${replay_kernel_file}
    )
    add_custom_target(replay_kernel_${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT} ALL
                      DEPENDS ${OPREPLAY_OUT_DIR}/${op_kerne_name}_entry_${OPREPLAY_COMPUTE_UNIT}.o
    )
    install(TARGETS replay_${op_kerne_name}_${OPREPLAY_COMPUTE_UNIT}
            LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_replay
    )
    install(FILES ${OPREPLAY_OUT_DIR}/${op_kerne_name}_entry_${OPREPLAY_COMPUTE_UNIT}.o
            DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_replay
    )
  endforeach()
endif()
endfunction()

function(add_npu_support_target)
cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
add_custom_command(OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
  COMMAND mkdir -p ${NPUSUP_OUT_DIR}
  COMMAND ${CMAKE_SOURCE_DIR}/cmake/util/gen_ops_filter.sh
          ${NPUSUP_OPS_INFO_DIR}
          ${NPUSUP_OUT_DIR}
)
add_custom_target(npu_supported_ops ALL
  DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json
)
install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
  DESTINATION ${NPUSUP_INSTALL_DIR}
)
endfunction()

function(add_bin_compile_target)
cmake_parse_arguments(BINCMP "" "TARGET;OPS_INFO;COMPUTE_UNIT;IMPL_DIR;ADP_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/src)
file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/bin)
file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/gen)
execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_bin_param_build.py
                        ${BINCMP_OPS_INFO} ${BINCMP_OUT_DIR}/gen ${BINCMP_COMPUTE_UNIT}
                RESULT_VARIABLE EXEC_RESULT
                OUTPUT_VARIABLE EXEC_INFO
                ERROR_VARIABLE  EXEC_ERROR
)
if (${EXEC_RESULT})
  message("ops binary compile scripts gen info: ${EXEC_INFO}")
  message("ops binary compile scripts gen error: ${EXEC_ERROR}")
  message(FATAL_ERROR "ops binary compile scripts gen failed!")
endif()
if (NOT TARGET binary)
  add_custom_target(binary)
endif()
add_custom_target(${BINCMP_TARGET}
                  COMMAND cp -r ${BINCMP_IMPL_DIR}/*.* ${BINCMP_OUT_DIR}/src
)
add_custom_target(${BINCMP_TARGET}_gen_ops_config
                  COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/insert_simplified_keys.py -p ${BINCMP_OUT_DIR}/bin
                  COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_ops_config.py -p ${BINCMP_OUT_DIR}/bin
                          -s ${BINCMP_COMPUTE_UNIT}
)
add_dependencies(binary ${BINCMP_TARGET}_gen_ops_config)
file(GLOB bin_scripts ${BINCMP_OUT_DIR}/gen/*.sh)
foreach(bin_script ${bin_scripts})
  get_filename_component(bin_file ${bin_script} NAME_WE)
  string(REPLACE "-" ";" bin_sep ${bin_file})
  list(GET bin_sep 0 op_type)
  list(GET bin_sep 1 op_file)
  list(GET bin_sep 2 op_index)
  if (NOT TARGET ${BINCMP_TARGET}_${op_file}_copy)
    file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/bin/${op_file})
    add_custom_target(${BINCMP_TARGET}_${op_file}_copy
                      COMMAND cp ${BINCMP_ADP_DIR}/${op_file}.py ${BINCMP_OUT_DIR}/src/${op_type}.py
    )
    install(DIRECTORY ${BINCMP_OUT_DIR}/bin/${op_file}
      DESTINATION ${BINCMP_INSTALL_DIR}/${BINCMP_COMPUTE_UNIT} OPTIONAL
    )
    install(FILES ${BINCMP_OUT_DIR}/bin/${op_file}.json
      DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}/ OPTIONAL
    )
  endif()
  add_custom_target(${BINCMP_TARGET}_${op_file}_${op_index}
                    COMMAND export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} && bash ${bin_script} ${BINCMP_OUT_DIR}/src/${op_type}.py ${BINCMP_OUT_DIR}/bin/${op_file}
                    WORKING_DIRECTORY ${BINCMP_OUT_DIR}
  )
  add_dependencies(${BINCMP_TARGET}_${op_file}_${op_index} ${BINCMP_TARGET} ${BINCMP_TARGET}_${op_file}_copy)
  add_dependencies(${BINCMP_TARGET}_gen_ops_config ${BINCMP_TARGET}_${op_file}_${op_index})
endforeach()
install(FILES ${BINCMP_OUT_DIR}/bin/binary_info_config.json
  DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT} OPTIONAL
)
endfunction()
