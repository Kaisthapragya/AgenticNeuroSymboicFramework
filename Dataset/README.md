# Dataset Title: T1D-UOM – A Longitudinal Multimodal Dataset of Type 1 Diabetes

[![DOI](https://zenodo.org/badge/884931751.svg)](https://doi.org/10.5281/zenodo.15169263)

## Overview

This dataset contains historical Type 1 Diabetes data collected across 16 individuals, covering activity data, blood glucose data, sleep data, insulin data, and nutrition Data. The data was collected in 2023 and 2024.

This dataset aims to support the development of algorithm in enahcning blood glucose management for people with Type 1 Diabetes Mellitus.

## Ethical approval

University of Manchester Ethical Approval 2024-15687-33719.

**Date Collected**: From October 2023 to August 2024

## Dataset Summary

Files inside each folder follow the naming convention:  
 `UoM + "Data category" + ParticipantID.csv`

### **Folders**

- Dataset
  - Demographics
    - Size: 4 KB
  - Activity Data
    - Size: 17.8 MB
  - Glucose Data
    - Size: 7.9 MB
  - Insulin Data
    - Basal Data
      - Size: 606 KB
    - Bolus Data
      - Size: 188 KB
  - Nutrition Data
    - Size: 295 KB
  - Sleep Data
    - Size: 13.9 MB
  - README.md

## **Number of Records**

| Data Type          | Files | Records |
| ------------------ | ----- | ------- |
| **Demographics**   | 1     | 16      |
| **Activity Data**  | 17    | 228,681 |
| **Glucose Data**   | 17    | 356,146 |
| **Basal Insulin**  | 14    | 20,407  |
| **Bolus Insulin**  | 16    | 5,660   |
| **Nutrition Data** | 15    | 4,351   |
| **Sleep Data**     | 30    | 323,340 |

## Data Format

- All files are in **CSV format** with **UTF-8 encoding**.
- Files use a comma (`","`) as the delimiter.

## **Data Dictionary**

### 'UoMBMI.csv'

| Column           | Type   | Description                    | Units | Possible Values |
| ---------------- | ------ | ------------------------------ | ----- | --------------- |
| `participant_id` | String | Participant ID                 | N/A   | N/A             |
| `weight_kg`      | Int    | Weight of participant          | kg    | N/A             |
| `height_m`       | Float  | Height of participant          | m     | N/A             |
| `bmi`            | Float  | Body Mass Index of participant | kg/m2 | N/A             |

---

### 'UoMActivityID.csv'

| Column                  | Type     | Description                  | Units               | Possible Values                       |
| ----------------------- | -------- | ---------------------------- | ------------------- | ------------------------------------- |
| `activity_ts`           | Datetime | Datetime of observation      | MM/DD/YYYY HH:MM:SS | N/A                                   |
| `activity_type`         | String   | Type of activity             | N/A                 | SEDENTARY, WALKING,RUNNING, & GENERIC |
| `active_Kcal`           | Int      | Calories burned actively     | kcal                | N/A                                   |
| `step_count`            | Int      | Steps taken                  | count               | N/A                                   |
| `distance_m`            | Float    | Distance covered             | meters              | N/A                                   |
| `duration_s`            | Int      | Duration of activity         | seconds             | N/A                                   |
| `active_time_s`         | Int      | Active time duration         | seconds             | N/A                                   |
| `start_time_s`          | Int      | Activity start time          | seconds             | N/A                                   |
| `start_time_offset_s`   | Int      | Start time offset            | seconds             | N/A                                   |
| `met`                   | Float    | Metabolic equivalent of task | METs                | N/A                                   |
| `intensity`             | String   | Intensity level              | N/A                 | SEDENTARY, ACTIVE & HIGHLY_ACTIVE     |
| `motion_intensity_mean` | Float    | Mean motion intensity        | N/A                 | N/A                                   |
| `motion_intensity_max`  | Float    | Maximum motion intensity     | N/A                 | N/A                                   |

---

### **UoMBasalID.csv**

| Column         | Type     | Description             | Units            | Possible Values            |
| -------------- | -------- | ----------------------- | ---------------- | -------------------------- |
| `basal_ts`     | Datetime | Datetime of observation | MM/DD/YYYY HH:MM | N/A                        |
| `basal_dose`   | Float    | Basal rate              | U or U/h         | N/A                        |
| `insulin_kind` | String   | Kind of insulin         | N/A              | R (Rapid), L (Long-acting) |

---

### **UoMBolusID.csv**

| Column       | Type     | Description             | Units            | Possible Values |
| ------------ | -------- | ----------------------- | ---------------- | --------------- |
| `bolus_ts`   | Datetime | Datetime of observation | MM/DD/YYYY HH:MM | N/A             |
| `bolus_dose` | Float    | Bolus dose              | U                | N/A             |

---

### **UoMGlucoseID.csv**

| Column  | Type     | Description             | Units               | Possible Values |
| ------- | -------- | ----------------------- | ------------------- | --------------- |
| `bg_ts` | Datetime | Datetime of observation | MM/DD/YYYY HH:MM:SS | N/A             |
| `value` | Float    | Blood glucose reading   | mmol/L              | N/A             |

---

### **UoMNutritionID.csv**

| Column      | Type     | Description             | Units            | Possible Values                 |
| ----------- | -------- | ----------------------- | ---------------- | ------------------------------- |
| `meal_ts`   | Datetime | Datetime of observation | MM/DD/YYYY HH:MM | N/A                             |
| `meal_type` | String   | Meal Type               | N/A              | Breakfast, Lunch, Dinner, Snack |
| `meal_tag`  | String   | Meal Tag                | N/A              | N/A                             |
| `carbs_g`   | Int      | Carbohydrates eaten     | g                | N/A                             |
| `prot_g`    | Int      | Proteins eaten          | g                | N/A                             |
| `fat_g`     | Int      | Fat eaten               | g                | N/A                             |
| `fibre_g`   | Int      | Fibre eaten             | g                | N/A                             |

### 'UoMsleepID.csv'

| Column                            | Type     | Description                | Units                  | Possible Values |
| --------------------------------- | -------- | -------------------------- | ---------------------- | --------------- |
| `sleep_ts`                        | Datetime | Datetime of observation    | MM/DD/YYYY HH:MM:SS    | N/A             |
| `step_count`                      | Int      | Steps taken                | count                  | N/A             |
| `heart_rate`                      | Int      | Heart rate                 | beats_per_minute (bPm) | N/A             |
| `current_activity_type_intensity` | Int      | Current activity intensity | count                  | N/A             |
| `stress_level_value`              | Int      | Stress level               | scale                  | N/A             |
| `sleep_level`                     | Int      | Sleep/awake state          | 0/1                    | 0/1             |
| `resting_heart_rate`              | Int      | Resting heart rate         | bPm                    | N/A             |

### 'UoMIDsleeptime.csv'

| Column                          | Type     | Description                          | Units                    | Possible Values                                  |
| ------------------------------- | -------- | ------------------------------------ | ------------------------ | ------------------------------------------------ |
| `sleep_start_ts`                | Datetime | Sleep start time                     | MM/DD/YYYY HH:MM:SS      | N/A                                              |
| `sleep_end_ts`                  | Datetime | Sleep end time                       | MM/DD/YYYY HH:MM:SS      | N/A                                              |
| `calendar_date`                 | Datetime | Calendar date                        | MM/DD/YYYY HH:MM:SS      | N/A                                              |
| `sleep_window_confirm_type`     | String   | Sleep confirmation                   | N/A                      | ENHANCED_CONFIRMED_FINAL, UNCONFIRMED, OFF_WRIST |
| `deep_sleep_s`                  | Int      | Seconds in deep sleep stage          | count                    | N/A                                              |
| `light_sleep_s`                 | Int      | Seconds in light sleep stage         | count                    | N/A                                              |
| `rem_sleep_s`                   | Int      | Seconds in REM sleep stage           | count                    | N/A                                              |
| `awake_sleep_s`                 | Int      | Seconds in awake stage               | count                    | N/A                                              |
| `unmeasurable_s`                | Int      | Unmeasurable seconds                 | count                    | N/A                                              |
| `sleep_levels_map_deep`         | String   | Mapping of deep sleep levels         | Seconds (Unix Timestamp) | N/A                                              |
| `sleep_levels_map_light`        | String   | Mapping of light sleep levels        | Seconds (Unix Timestamp) | N/A                                              |
| `sleep_levels_map_awake`        | String   | Mapping of awake sleep levels        | Seconds (Unix Timestamp) | N/A                                              |
| `sleep_levels_map_rem`          | String   | Mapping of REM sleep levels          | Seconds (Unix Timestamp) | N/A                                              |
| `sleep_levels_map_unmeasurable` | String   | Mapping of unmeasurable sleep levels | Seconds (Unix Timestamp) | N/A                                              |
| `validation`                    | String   | Validation status of sleep data      | Seconds (Unix Timestamp) | ENHANCED_FINAL, ENHANCED_TENTATIVE               |
