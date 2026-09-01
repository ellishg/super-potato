idf_component_register(SRCS "main.c"
                       PRIV_REQUIRES spi_flash esp_partition
                       INCLUDE_DIRS "")

set(model_file "${PROJECT_BINARY_DIR}/model.bin")
set(generate_model_script "${PROJECT_DIR}/scripts/generate_model.py")
add_custom_command(
    OUTPUT "${model_file}"
    COMMAND ${PYTHON} "${generate_model_script}" "${model_file}"
    DEPENDS "${generate_model_script}"
    COMMENT "Generating ${model_file}"
)
add_custom_target(generate_model_file ALL DEPENDS "${model_file}")
esp_partition_register_target("model" "${model_file}" DEPENDS generate_model_file FLASH_IN_PROJECT)
