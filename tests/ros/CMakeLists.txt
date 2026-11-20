add_executable(rko_lio_ros_tests test_point_cloud_read.cpp)

target_link_libraries(rko_lio_ros_tests PRIVATE rko_lio::ros::utils Catch2::Catch2WithMain)

target_compile_features(rko_lio_ros_tests PRIVATE cxx_std_20)

catch_discover_tests(rko_lio_ros_tests)
