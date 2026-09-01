#include "esp_check.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_system.h"
#include <stdio.h>

#define BYTES_TO_KB(bytes) ((uint32_t)((bytes) / 1024))
#define BYTES_TO_MB(bytes) ((uint32_t)((bytes) / (1024 * 1024)))

static const char *TAG = "super-potato";

esp_err_t run(void) {
  esp_chip_info_t chip_info;
  esp_chip_info(&chip_info);

  uint32_t flash_size;
  ESP_RETURN_ON_ERROR(esp_flash_get_size(NULL, &flash_size), TAG,
                      "Get flash size failed");

  ESP_LOGI(TAG, "%" PRIu32 "MB %s flash", BYTES_TO_MB(flash_size),
           (chip_info.features & CHIP_FEATURE_EMB_FLASH) ? "embedded"
                                                         : "external");
  ESP_LOGI(TAG, "Minimum free heap size: %" PRIu32 " KB\n",
           BYTES_TO_KB(esp_get_minimum_free_heap_size()));

  const esp_partition_t *partition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
  ESP_RETURN_ON_FALSE(partition, ESP_ERR_NOT_FOUND, TAG, "Partition not found");

  esp_partition_mmap_handle_t map_handle;
  const char *model_data;
  ESP_RETURN_ON_ERROR(
      esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA,
                         (const void **)&model_data, &map_handle),
      TAG, "Failed to mmap model");
  ESP_LOGI(TAG, "model mmaped at %p with size %" PRIu32 " MB", model_data,
           BYTES_TO_MB(partition->size));

  for (size_t i = 0; i < 100; i++) {
    printf("%c", model_data[i]);
  }
  return ESP_OK;
}

void app_main(void) {
  esp_err_t err = run();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Error running application: %s", esp_err_to_name(err));
  }
  fflush(stdout);
  ESP_LOGI(TAG, "Application finished");
}