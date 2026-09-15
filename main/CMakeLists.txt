include(FetchContent)

if(NOT CMAKE_SCRIPT_MODE_FILE)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_Declare(
        utf8proc_source
        GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc.git
        GIT_TAG v2.9.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(utf8proc_source)
endif()

idf_component_register(SRCS "main.c"
                       PRIV_REQUIRES spi_flash esp_partition esp_timer esp_psram
                       INCLUDE_DIRS "")

if(NOT CMAKE_SCRIPT_MODE_FILE)
    target_link_libraries(${COMPONENT_LIB} PRIVATE utf8proc)
endif()

# set_source_files_properties(
#     "${CMAKE_CURRENT_LIST_DIR}/main.c"
#     PROPERTIES COMPILE_OPTIONS "-ffast-math"
# )
