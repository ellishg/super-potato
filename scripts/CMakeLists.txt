set(model_file "${PROJECT_BINARY_DIR}/model.bin")
set(tokenizer_file "${PROJECT_BINARY_DIR}/tokenizer.bin")
set(generate_model_script "${CMAKE_CURRENT_SOURCE_DIR}/generate_model.sh")

add_custom_command(
    OUTPUT "${model_file}" "${tokenizer_file}"
    COMMAND ${BASH} "${generate_model_script}" "${model_file}" "${tokenizer_file}"
    DEPENDS "${generate_model_script}"
    COMMENT "Generating model and tokenizer files"
)

add_custom_target(generate_model_files ALL DEPENDS "${model_file}" "${tokenizer_file}")

esp_partition_register_target("model" "${model_file}" DEPENDS generate_model_files FLASH_IN_PROJECT)
esp_partition_register_target("tokenizer" "${tokenizer_file}" DEPENDS generate_model_files FLASH_IN_PROJECT)
