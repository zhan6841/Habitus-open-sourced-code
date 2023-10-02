# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("habitus")

add_compile_options(-Wall -Wextra)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

find_library(android-lib android)
find_library(GLESv2-lib GLESv2)
find_library(GLESv3-lib GLESv3)

include_directories(/home/anlan/AndroidStudioProjects/Habitus/libpcap/install/include/)

add_library(libpcap STATIC IMPORTED)
set_target_properties(libpcap
        PROPERTIES IMPORTED_LOCATION
        /home/anlan/AndroidStudioProjects/Habitus/libpcap/install/lib/libpcap.a
        )

# # Import the ARCore (Google Play Services for AR) library.
# add_library(arcore SHARED IMPORTED)
# set_target_properties(arcore PROPERTIES IMPORTED_LOCATION
#         /home/anlan/AndroidStudioProjects/Habitus/app/build/arcore-native/jni/${ANDROID_ABI}/libarcore_sdk_c.so
#         INTERFACE_INCLUDE_DIRECTORIES /home/anlan/AndroidStudioProjects/Habitus/app/libraries/arcore-native/include
#         )

# # Import the glm header file from the NDK.
# add_library( glm INTERFACE )
# set_target_properties( glm PROPERTIES INTERFACE_INCLUDE_DIRECTORIES /home/anlan/AndroidStudioProjects/Habitus/third_party/glm)

# # Creates and names a library, sets it as either STATIC
# # or SHARED, and provides the relative paths to its source code.
# # You can define multiple libraries, and CMake builds them for you.
# # Gradle automatically packages shared libraries with your APK.

# ####################################################################
# ##################### Script for habitus ###########################
# ####################################################################
# add_library( # Sets the name of the library.
#         habitus

#         # Sets the library as a shared library.
#         SHARED

#         # Provides a relative path to your source file(s).
#         native-lib.cpp
#         habitus.h
#         tools.h tools.cpp
#         settings.h settings.cpp
#         feature_collection.h feature_collection.cpp
#         tput_estimator.h tput_estimator.cpp
#         senor_collector.h senor_collector.cpp
#         video_msg.h video_msg.cpp
#         video_comm.h video_comm.cpp
#         video_data.h video_data.cpp
#         video_scheduler.h video_scheduler.cpp
#         frame_buffer.h frame_buffer.cpp
#         hmd_tracker.h hmd_tracker.cpp
#         sys_perf.h sys_perf.cpp
#         ar_player/ar_player.h ar_player/ar_player.cc
#         ar_player/background_renderer.h ar_player/background_renderer.cc
#         ar_player/util.h ar_player/util.cc
#         ar_player/glm.h)

# # Specifies libraries CMake should link to your target library. You
# # can link multiple libraries, such as libraries you define in this
# # build script, prebuilt third-party libraries, or system libraries.

# target_link_libraries( # Specifies the target library.
#         habitus

#         libpcap

#         # Links the target library to the log library
#         # included in the NDK.
#         ${log-lib}
#         ${android-lib}
#         ${GLESv2-lib}
#         glm
#         arcore)

# ####################################################################
# ############ Script for feature_collection_service #################
# ####################################################################

# add_executable(feature_collection_service
#         feature_collection_service.cpp
#         habitus.h
#         tools.h tools.cpp
#         tput_estimator.h tput_estimator.cpp)

# target_link_libraries(feature_collection_service libpcap ${log-lib})

# ####################################################################
# ################# Script for data_downloader #######################
# ####################################################################

# add_executable(data_downloader download_test.cpp)

# ####################################################################
# ################## Script for tput_reporter ########################
# ####################################################################

# add_executable(tput_reporter
#         tput_reporter.cpp
#         habitus.h
#         tools.h tools.cpp
#         tput_estimator.h tput_estimator.cpp
#         senor_collector.h senor_collector.cpp)

# target_link_libraries(tput_reporter libpcap ${log-lib})

####################################################################
############## Script for data_collector_android ###################
####################################################################

add_executable(data_collector_android
        data_collector_android.cpp)

target_link_libraries(data_collector_android libpcap)