# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16)

project(cann_ops_adv)

option(BUILD_OPEN_PROJECT         "Build open ascend ops project."  ON)
option(ENABLE_CCACHE              "Enable ccache capability"        ON)
set(ASCEND_COMPUTE_UNIT           "ascend910b"                    CACHE   STRING   "soc that need to be compiled")
set(ASCEND_OP_NAME                "ALL"                           CACHE   STRING   "operators that need to be compiled")
set(VENDOR_NAME                   "customize"                     CACHE   STRING   "vendor name")

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)


set(_op_host_aclnn_link
        $<BUILD_INTERFACE:intf_pub>
        exe_graph
        register
        tiling_api
        c_sec
)

add_library(op_host_aclnn SHARED EXCLUDE_FROM_ALL)
target_link_libraries(op_host_aclnn PRIVATE
        ${_op_host_aclnn_link}
)
target_compile_options(op_host_aclnn PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
)

add_library(op_host_aclnnInner SHARED EXCLUDE_FROM_ALL)
target_link_libraries(op_host_aclnnInner PRIVATE
        ${_op_host_aclnn_link}
)
target_compile_options(op_host_aclnnInner PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
)

add_library(op_host_aclnnExc SHARED EXCLUDE_FROM_ALL)
target_link_libraries(op_host_aclnnExc PRIVATE
        ${_op_host_aclnn_link}
)
target_compile_options(op_host_aclnnExc PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
)

# op api
add_library(opapi SHARED)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
                COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
        )

        target_sources(opapi PRIVATE
                ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
        )
target_compile_options(opapi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
)
target_include_directories(opapi PRIVATE
        $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
        $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/aclnn>
        # $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/aclnn_kernels>
)
target_compile_options(opapi PRIVATE
        -Werror=format
)
target_compile_definitions(opapi PRIVATE
        -DACLNN_LOG_FMT_CHECK
)
target_link_libraries(opapi PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        -Wl,--whole-archive
        ops_aclnn
        -Wl,--no-whole-archive
        -lopapi
        nnopbase
        profapi
        ge_common_base
        ascend_dump
        ascendalog
        dl
)
set_target_properties(opapi PROPERTIES OUTPUT_NAME
        cust_opapi
)
install(TARGETS opapi
        LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}/op_api/lib
)

# op proto
add_library(opsproto SHARED)
file(GLOB PROTO_AUTO_FILES
       ${ASCEND_AUTOGEN_DIR}/*_proto.cpp
    )
target_sources(opsproto PRIVATE
       ${PROTO_AUTO_FILES}
)
target_compile_options(opsproto PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
        -fvisibility=hidden
)
target_compile_definitions(opsproto PRIVATE
        LOG_CPP
        PROCESS_LOG
        OP_PROTO_LIB
)
target_link_libraries(opsproto PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:ops_utils_proto_headers>
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
        -Wl,--no-as-needed
        exe_graph
        graph
        graph_base
        register
        ascendalog
        error_manager
        platform
        -Wl,--as-needed
        c_sec
)
set_target_properties(opsproto PROPERTIES OUTPUT_NAME
        cust_opsproto_rt2.0
)
install(TARGETS opsproto
        LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
)

# op tiling
add_library(optiling SHARED)
target_sources(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/src/fallback_comm.cpp
)
target_compile_options(optiling PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
        -fvisibility=hidden
)
target_compile_definitions(optiling PRIVATE
        LOG_CPP
        PROCESS_LOG
        OP_TILING_LIB
)
target_link_libraries(optiling PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:ops_utils_tiling_headers>
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
        -Wl,--no-as-needed
        graph
        graph_base
        exe_graph
        platform
        register
        ascendalog
        error_manager
        -Wl,--as-needed
        -Wl,--whole-archive
        tiling_api
        -Wl,--no-whole-archive
        mmpa
        c_sec
)
set_target_properties(optiling PROPERTIES OUTPUT_NAME
        cust_opmaster_rt2.0
)
add_custom_command(TARGET optiling
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TILING_CUSTOM_DIR}
        COMMAND ln -sf $<TARGET_FILE:optiling> ${TILING_CUSTOM_FILE}
)
install(TARGETS optiling
        LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
)

# optiling compat
set(compat_optiling_dir  ${CMAKE_CURRENT_BINARY_DIR}/compat)
set(compat_optiling_file ${compat_optiling_dir}/liboptiling.so)
add_custom_target(optiling_compat ALL
        DEPENDS ${compat_optiling_file}
)

add_custom_command(
        OUTPUT ${compat_optiling_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${compat_optiling_dir}
        COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:optiling> ${compat_optiling_file}
)

install(FILES ${compat_optiling_file}
        DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/op_tiling
)

add_ops_tiling_keys(
        OP_NAME "ALL"
        TILING_KEYS ${TILING_KEY}
)

add_opc_config(
        OP_NAME "ALL"
        CONFIG ${OP_DEBUG_CONFIG}
)

if(ADD_OPS_COMPILE_OPTION_V2)
add_ops_compile_options(
        OP_NAME "ALL"
        OPTIONS ${OPS_COMPILE_OPTIONS}
)
endif()


add_subdirectory(src/common)
add_subdirectory(src/rwkv7)
add_subdirectory(src/rwkv6)
# add_subdirectory(src/rwkv4)
# add_subdirectory(tests)
if (TESTS_EXAMPLE_OPS_TEST)
add_subdirectory(examples)
endif ()

set(OP_LIST)
set(OP_DIR_LIST)
op_add_subdirectory(OP_LIST OP_DIR_LIST)

foreach (OP_DIR ${OP_DIR_LIST})
    add_subdirectory(${OP_DIR})
endforeach ()

set(OP_DEPEND_DIR_LIST)
op_add_depend_directory(
        OP_LIST ${OP_LIST}
        OP_DIR_LIST OP_DEPEND_DIR_LIST
)
foreach (OP_DEPEND_DIR ${OP_DEPEND_DIR_LIST})
    add_subdirectory(${OP_DEPEND_DIR})
endforeach ()

# ------------------------------------------------ aclnn ------------------------------------------------
get_target_property(base_aclnn_srcs op_host_aclnn SOURCES)
get_target_property(base_aclnn_inner_srcs op_host_aclnnInner SOURCES)
get_target_property(base_aclnn_exclude_srcs op_host_aclnnExc SOURCES)


set(base_aclnn_binary_dir ${ASCEND_AUTOGEN_DIR})


set(generate_aclnn_srcs)
set(generate_aclnn_inner_srcs)
set(generate_aclnn_headers)
set(generate_proto_dir ${base_aclnn_binary_dir})
set(generate_exclude_proto_srcs)
set(generate_proto_srcs)
set(generate_proto_headers)

if (base_aclnn_srcs)
    foreach (_src ${base_aclnn_srcs})
        string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
        if (is_match)
            get_filename_component(name_without_ext ${_src} NAME_WE)

            string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
            list(APPEND generate_aclnn_srcs ${base_aclnn_binary_dir}/aclnn_${_op_name}.cpp)
            list(APPEND generate_aclnn_headers ${base_aclnn_binary_dir}/aclnn_${_op_name}.h)
            list(APPEND generate_proto_srcs    ${generate_proto_dir}/${_op_name}_proto.cpp)
            list(APPEND generate_proto_headers ${generate_proto_dir}/${_op_name}_proto.h)
        endif ()
    endforeach ()
else ()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
    )

    target_sources(op_host_aclnn PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
    )
endif ()

if (base_aclnn_inner_srcs)
    foreach (_src ${base_aclnn_inner_srcs})
        string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
        if (is_match)
            get_filename_component(name_without_ext ${_src} NAME_WE)
            string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
            list(APPEND generate_aclnn_inner_srcs ${base_aclnn_binary_dir}/inner/aclnnInner_${_op_name}.cpp)
            list(APPEND generate_proto_srcs    ${generate_proto_dir}/inner/${_op_name}_proto.cpp)
            list(APPEND generate_proto_headers ${generate_proto_dir}/inner/${_op_name}_proto.h)
        endif ()
    endforeach ()
else ()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
    )

    target_sources(op_host_aclnnInner PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
    )
endif ()

if (base_aclnn_exclude_srcs)
    foreach (_src ${base_aclnn_exclude_srcs})
        string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
        if (is_match)
            get_filename_component(name_without_ext ${_src} NAME_WE)
            string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
            list(APPEND generate_exclude_proto_srcs    ${generate_proto_dir}/exc/${_op_name}_proto.cpp)
            list(APPEND generate_proto_srcs            ${generate_proto_dir}/exc/${_op_name}_proto.cpp)
            list(APPEND generate_proto_headers         ${generate_proto_dir}/exc/${_op_name}_proto.h)
        endif ()
    endforeach ()
else()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
    )

    target_sources(op_host_aclnnExc PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
    )
endif ()

if (generate_aclnn_srcs OR generate_aclnn_inner_srcs)
        set(ops_aclnn_src ${generate_aclnn_srcs} ${generate_aclnn_inner_srcs})
else ()
        set(ops_aclnn_src ${CMAKE_CURRENT_BINARY_DIR}/ops_aclnn_src_stub.cpp)

        add_custom_command(OUTPUT ${ops_aclnn_src}
                COMMAND touch ${ops_aclnn_src}
        )
endif ()

set_source_files_properties(${ops_aclnn_src}
        PROPERTIES GENERATED TRUE
)
add_library(ops_aclnn STATIC
        ${ops_aclnn_src}
)
target_compile_options(ops_aclnn PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
)
target_link_libraries(ops_aclnn PRIVATE
        $<BUILD_INTERFACE:intf_pub>
)
add_dependencies(ops_aclnn opbuild_gen_default opbuild_gen_inner)

set_source_files_properties(${generate_proto_srcs}
        PROPERTIES GENERATED TRUE
)
target_sources(opsproto PRIVATE
        ${generate_proto_srcs}
)
add_dependencies(opsproto ops_proto_headers)

install(FILES ${generate_proto_headers}
        DESTINATION packages/vendors/${VENDOR_NAME}/op_proto/inc OPTIONAL
)

redefine_file_macro(
        TARGET_NAME
        op_host_aclnn
        op_host_aclnnInner
        op_host_aclnnExc
        opapi
        opsproto
        optiling
        ops_aclnn
)


if (generate_aclnn_headers)
    install(FILES ${generate_aclnn_headers}
            DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
    )
endif ()

add_library(ops_proto_headers INTERFACE)

target_include_directories(ops_proto_headers INTERFACE
        $<BUILD_INTERFACE:${generate_proto_dir}>
        $<BUILD_INTERFACE:${generate_proto_dir}/inner>
        $<BUILD_INTERFACE:${generate_proto_dir}/exc>
        $<INSTALL_INTERFACE:include/ops_adv/proto>
)


add_dependencies(ops_proto_headers opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)


# ------------------------------------------------ opbuild ------------------------------------------------
if (generate_aclnn_srcs)
add_custom_command(OUTPUT ${generate_aclnn_srcs} ${generate_aclnn_headers}
        COMMAND mkdir -p ${base_aclnn_binary_dir}
        COMMAND OPS_PROTO_SEPARATE=1
        OPS_ACLNN_GEN=1
        OPS_PROJECT_NAME=aclnn
        ${OP_BUILD_TOOL}
        $<TARGET_FILE:op_host_aclnn>
        ${base_aclnn_binary_dir}
)
endif ()

add_custom_target(opbuild_gen_default
        DEPENDS ${generate_aclnn_srcs} ${generate_aclnn_headers} op_host_aclnn
)

if (generate_aclnn_inner_srcs)
add_custom_command(OUTPUT ${generate_aclnn_inner_srcs}
        COMMAND mkdir -p ${base_aclnn_binary_dir}/inner
        COMMAND OPS_PROTO_SEPARATE=1
        OPS_ACLNN_GEN=1
        OPS_PROJECT_NAME=aclnnInner
        ${OP_BUILD_TOOL}
        $<TARGET_FILE:op_host_aclnnInner>
        ${base_aclnn_binary_dir}/inner
)
endif ()

add_custom_target(opbuild_gen_inner
        DEPENDS ${generate_aclnn_inner_srcs} op_host_aclnnInner
)

if (generate_exclude_proto_srcs)
add_custom_command(OUTPUT ${generate_exclude_proto_srcs}
        COMMAND mkdir -p ${base_aclnn_binary_dir}/exc
        COMMAND OPS_PROTO_SEPARATE=1
        OPS_ACLNN_GEN=0
        OPS_PROJECT_NAME=aclnnExc
        ${OP_BUILD_TOOL}
        $<TARGET_FILE:op_host_aclnnExc>
        ${base_aclnn_binary_dir}/exc
)
endif ()

add_custom_target(opbuild_gen_exc
        DEPENDS ${generate_exclude_proto_srcs} op_host_aclnnExc
)


# ------------------------------------------------ generate adapt py ------------------------------------------------
add_custom_target(generate_adapt_py
        COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_impl_build.py
        \"\"
        \"\"
        \"\"
        \"\"
        ${ASCEND_IMPL_OUT_DIR}
        ${ASCEND_AUTOGEN_DIR}
        --opsinfo-dir ${base_aclnn_binary_dir} ${base_aclnn_binary_dir}/inner ${base_aclnn_binary_dir}/exc
)

add_dependencies(generate_adapt_py opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)

foreach (_op_name ${OP_LIST})
    install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}.py
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
            OPTIONAL
    )
endforeach ()

foreach (_op_name ${OP_LIST})
    install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}.cpp
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
            OPTIONAL
    )
endforeach ()

install(DIRECTORY ${OPS_ADV_UTILS_KERNEL_INC}/
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/common
)

foreach (op_dir ${OP_DIR_LIST})
    get_filename_component(_op_name "${op_dir}" NAME)

    file(GLOB KERNEL_FILES
            ${op_dir}/op_kernel/*.cpp
            ${op_dir}/op_kernel/*.h
    )

    install(FILES ${KERNEL_FILES}
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
            OPTIONAL
    )
endforeach ()

# ------------------------------------------------ generate compile cmd ------------------------------------------------
add_custom_target(prepare_build ALL)
add_custom_target(generate_compile_cmd ALL)
add_custom_target(generate_ops_info ALL)
add_dependencies(prepare_build generate_adapt_py generate_compile_cmd)

foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
add_compile_cmd_target(
        COMPUTE_UNIT ${compute_unit}
)

add_ops_info_target(
        COMPUTE_UNIT ${compute_unit}
)
endforeach ()


# ------------------------------------------------ opp kernel ------------------------------------------------
if (ENABLE_OPS_KERNEL)
    add_custom_target(ops_kernel ALL)
    add_custom_target(ops_config ALL)
    add_dependencies(ops_kernel ops_config)

    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        add_bin_compile_target(
                COMPUTE_UNIT
                ${compute_unit}
                OP_INFO
                ${OP_DIR_LIST}
        )
    endforeach ()
endif ()

add_custom_target(modify_vendor ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/scripts/install.sh ${CMAKE_CURRENT_BINARY_DIR}/scripts/upgrade.sh
)

# modify VENDOR_NAME in install.sh and upgrade.sh
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scripts/install.sh ${CMAKE_CURRENT_BINARY_DIR}/scripts/upgrade.sh
        COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/scripts
        COMMAND cp -r ${ASCEND_PROJECT_DIR}/scripts/* ${CMAKE_CURRENT_BINARY_DIR}/scripts/
        COMMAND chmod +w ${CMAKE_CURRENT_BINARY_DIR}/scripts/*
        COMMAND sed -i "s/vendor_name=customize/vendor_name=${VENDOR_NAME}/g" ${CMAKE_CURRENT_BINARY_DIR}/scripts/*
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/
        DESTINATION . FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
)

# gen version.info
set(version_info_dir  ${CMAKE_CURRENT_BINARY_DIR})
set(version_info_file ${version_info_dir}/version.info)
add_custom_target(gen_version_info ALL
        DEPENDS ${version_info_file}
)

add_custom_command(OUTPUT ${version_info_file}
        COMMAND bash ${ASCENDC_CMAKE_UTIL_DIR}/gen_version_info.sh ${ASCEND_CANN_PACKAGE_PATH} ${version_info_dir}
)

install(FILES ${version_info_file}
        DESTINATION packages/vendors/${VENDOR_NAME}/
)

# CPack config
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION "CPack ops project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack ops project")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
set(CPACK_PACKAGE_FILE_NAME "CANN-custom_ops-${CANN_VERSION}-linux.${CMAKE_SYSTEM_PROCESSOR}.run")
set(CPACK_GENERATOR External)
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${ASCEND_CMAKE_DIR}/makeself.cmake)
set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
include(CPack)


# generate npu_supported_ops.json
add_npu_support_target(TARGET npu_supported_ops
        OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
        OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
        INSTALL_DIR packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE}
)