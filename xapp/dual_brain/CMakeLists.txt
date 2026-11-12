# Harmony Dual-Brain congestion-control xApp (needs ONNX Runtime C API + cJSON).
# Skipped (with a message) when either is missing, so stock FlexRIC builds are unaffected.
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime C release root (include/, lib/)")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h PATHS ${ONNXRUNTIME_ROOT}/include NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
find_path(CJSON_INCLUDE_DIR cjson/cJSON.h)
find_library(CJSON_LIB cjson)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB OR NOT CJSON_INCLUDE_DIR OR NOT CJSON_LIB)
  message(STATUS "dual_brain xApp: skipped (ONNX Runtime at ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT} and libcjson-dev required)")
  return()
endif()

add_library(harmony_cc_model STATIC cc_model.c)
target_include_directories(harmony_cc_model PUBLIC ${ONNXRUNTIME_INCLUDE_DIR} ${CJSON_INCLUDE_DIR})
target_link_libraries(harmony_cc_model PUBLIC ${ONNXRUNTIME_LIB} ${CJSON_LIB} m)
set_target_properties(harmony_cc_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(xapp_congestion_ctrl
               xapp_congestion_ctrl.c
               ../../../../src/util/alg_ds/alg/defer.c
               ../../../../src/util/time_now_us.c)
target_link_libraries(xapp_congestion_ctrl PUBLIC e42_xapp harmony_cc_model -pthread -lsctp -ldl)
set_target_properties(xapp_congestion_ctrl PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)

# Parity test: C model path vs NeuralSmith predictions on holdout.csv.
add_executable(cc_assoc_test cc_assoc_test.c)
add_executable(cc_policy_test cc_policy_test.c)
add_executable(cc_model_test cc_model_test.c)
target_link_libraries(cc_model_test PRIVATE harmony_cc_model)
set_target_properties(cc_model_test PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)

# R1 parity replay: recorded windows through the xApp's feature + model path (r1/zto/parity.py).
add_executable(cc_replay cc_replay.c)
target_link_libraries(cc_replay PRIVATE harmony_cc_model)
set_target_properties(cc_replay PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)
