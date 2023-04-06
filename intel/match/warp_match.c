#include <cstdio>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <level_zero/ze_api.h>
#include "../include/ze_utils.h"

int NUM_ROUNDS = 1;

int main(int argc, char* argv[]) {
    if (argc < 3)
       exit_msg(strcat(argv[0], " groupSize numGroup" ));

    ze_result_t errno;

    // driver and device discovery
    // Initialize the driver

    //  PLATFORM AND DEVICE
    printf(">>> Initializing OpenCL Platform and Device...\n");
    errno  = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    check_error(errno, "zeInit");

    // Discover all the driver instances
    uint32_t driverCount = 0;
    errno = zeDriverGet(&driverCount, NULL);
    check_error(errno, "zeDriverGet");

    //Now where the phDrivers
    ze_driver_handle_t* phDrivers = (ze_driver_handle_t*) malloc(driverCount * sizeof(ze_driver_handle_t));
    errno = zeDriverGet(&driverCount, phDrivers);
    check_error(errno, "zeDriverGet");

    // Device who will be selected
    ze_device_handle_t hDevice = NULL;
    ze_driver_handle_t hDriver = NULL;

    for(uint32_t driver_idx = 0; driver_idx < driverCount; driver_idx++) {

        hDriver = phDrivers[driver_idx];
        /* - - - -
        Device
        - - - - */

        // if count is zero, then the driver will update the value with the total number of devices available.
        uint32_t deviceCount = 0;
        errno = zeDeviceGet(hDriver, &deviceCount, NULL);
        check_error(errno, "zeDeviceGet");

        ze_device_handle_t* phDevices = (ze_device_handle_t*) malloc(deviceCount * sizeof(ze_device_handle_t));
        errno = zeDeviceGet(hDriver, &deviceCount, phDevices);
        check_error(errno, "zeDeviceGet");

        for(uint32_t device_idx = 0;  device_idx < deviceCount; device_idx++) {
            ze_device_properties_t device_properties;
            errno = zeDeviceGetProperties(phDevices[device_idx], &device_properties);
            check_error(errno, "zeDeviceGetProperties");
            if (device_properties.type == ZE_DEVICE_TYPE_GPU){
                    printf("Running on Device #%d %s who is a GPU. \n", device_idx, device_properties.name);
                    hDevice = phDevices[device_idx];
                    break;
            }
        }

        free(phDevices);
        if (hDevice != NULL) {
            break;
        }
    }

    free(phDrivers);


    // CONTEXT
    ze_context_handle_t hContext = NULL;
    // Create context
    ze_context_desc_t context_desc = {
        ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        NULL,
        0};
    errno = zeContextCreate(hDriver, &context_desc, &hContext);
    check_error(errno, "zeContextCreate");


    // COMMAND QUEUE
    // Discover all command queue types
    uint32_t cmdqueueGroupCount = 0;
    zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, NULL);

    ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*) malloc(cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
    errno = zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, cmdqueueGroupProperties);
    check_error(errno, "zeDeviceGetCommandQueueGroupProperties");

    // Find a proper command queue
    uint32_t computeQueueGroupOrdinal = cmdqueueGroupCount;
    for( uint32_t i = 0; i < cmdqueueGroupCount; ++i ) {
        if( cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
            computeQueueGroupOrdinal = i;
            break;
        }
    }

    // Command queue
    ze_command_queue_desc_t commandQueueDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        NULL,
        computeQueueGroupOrdinal,
        0, // index
        0, // flags
        ZE_COMMAND_QUEUE_MODE_DEFAULT,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };

    ze_command_queue_handle_t hCommandQueue;
    errno = zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue);
    check_error(errno, "zeCommandQueueCreate");


    // COMMAND LIST
    ze_command_list_desc_t commandListDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        NULL,
        computeQueueGroupOrdinal,
        0 // flags
    };

    ze_command_list_handle_t hCommandList;
    errno =  zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList);
    check_error(errno, "zeCommandListCreate");


    // MODULE
    unsigned char* program_file_all; size_t program_size_all;
    errno = read_from_binary(&program_file_all, &program_size_all, "match_all_sync.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_any; size_t program_size_any;
    errno = read_from_binary(&program_file_any, &program_size_any, "match_any_sync.ar");
    check_error(errno, "read_from_binary");

    // OpenCL C kernel has been compiled to Gen Binary
    ze_module_desc_t moduleDescAll = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_all,
        program_file_all,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleAll;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescAll, &hModuleAll, NULL);

    ze_module_desc_t moduleDescAny = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_any,
        program_file_any,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleAny;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescAny, &hModuleAny, NULL);


    // KERNEL
    ze_kernel_desc_t kernelDescAllTrue = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_match_all_sync_custom_true"
    };

    ze_kernel_desc_t kernelDescAllFalse = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_match_all_sync_custom_false"
    };

    ze_kernel_desc_t kernelDescAnySimple = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_match_any_sync_custom_simple"
    };

    ze_kernel_desc_t kernelDescAnyAlternate = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_match_any_sync_custom_alternate"
    };

    ze_kernel_desc_t kernelDescAnyUnique = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_match_any_sync_custom_unique"
    };

    uint32_t groupSizeX =  (uint32_t) atoi(argv[1]);
    uint32_t numGroupsX =  (uint32_t) atoi(argv[2]);
    ze_kernel_handle_t hKernelAllTrue, hKernelAllFalse, hKernelAnySimple, hKernelAnyAlternate, hKernelAnyUnique;


    int match_all_sync_shared_var_arr_true[32];
    int match_all_sync_updated_true[32] = {0};

    int match_all_sync_shared_var_arr_false[32];
    int match_all_sync_updated_false[32] = {0};

    int match_any_sync_shared_var_arr_simple[32];
    int match_any_sync_updated_simple[32] = {0};

    int match_any_sync_shared_var_arr_alternate[32];
    int match_any_sync_updated_alternate[32] = {0};

    int match_any_sync_shared_var_arr_unique[32];
    int match_any_sync_updated_unique[32] = {0};

    errno = zeKernelCreate(hModuleAll, &kernelDescAllTrue, &hKernelAllTrue);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAll, &kernelDescAllFalse, &hKernelAllFalse);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAny, &kernelDescAnySimple, &hKernelAnySimple);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAny, &kernelDescAnyAlternate, &hKernelAnyAlternate);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAny, &kernelDescAnyUnique, &hKernelAnyUnique);
    check_error(errno, "zeKernelCreate");

    ze_group_count_t launchArgs = { numGroupsX, 1, 1 };
    srand(time(0));
    int arg = rand() % 1000;
    // Append launch kernel
    zeKernelSetArgumentValue(hKernelAllTrue, 0, sizeof(int), arg);
    zeKernelSetArgumentValue(hKernelAllTrue, 1, sizeof(int)*32, match_all_sync_shared_var_arr_true);
    zeKernelSetArgumentValue(hKernelAllTrue, 2, sizeof(int)*32, match_all_sync_updated_true);
    zeKernelSetGroupSize(hKernelAllTrue, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelAllFalse, 0, sizeof(int)*32, match_all_sync_shared_var_arr_false);
    zeKernelSetArgumentValue(hKernelAllFalse, 1, sizeof(int)*32, match_all_sync_updated_false);
    zeKernelSetGroupSize(hKernelAllFalse, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelAnySimple, 0, sizeof(int)*32, match_any_sync_shared_var_arr_simple);
    zeKernelSetArgumentValue(hKernelAnySimple, 1, sizeof(int)*32, match_any_sync_updated_simple);
    zeKernelSetGroupSize(hKernelAnySimple, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelAnyAlternate, 0, sizeof(int)*32, match_any_sync_shared_var_arr_alternate);
    zeKernelSetArgumentValue(hKernelAnyAlternate, 1, sizeof(int)*32, match_any_sync_updated_alternate);
    zeKernelSetGroupSize(hKernelAnyAlternate, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelAnyUnique, 0, sizeof(int)*32, match_any_sync_shared_var_arr_unique);
    zeKernelSetArgumentValue(hKernelAnyUnique, 1, sizeof(int)*32, match_any_sync_updated_unique);
    zeKernelSetGroupSize(hKernelAnyUnique, groupSizeX, 1, 1);

    zeCommandListAppendLaunchKernel(hCommandList, hKernelAllTrue, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelAllFalse, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelAnySimple, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelAnyAlternate, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelAnyUnique, &launchArgs, NULL, 0, NULL);

    // finished appending commands (typically done on another thread)
    errno = zeCommandListClose(hCommandList);
    check_error(errno, "zeCommandListClose");

    // SUBMISSION    
    
    // Execute command list in command queue
    double total_custom_time = 0;
    struct timeval start_custom, end_custom;
    gettimeofday(&start_custom, 0);

    errno = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, NULL);
    check_error(errno, "zeCommandQueueExecuteCommandLists");

    // synchronize host and device
    errno = zeCommandQueueSynchronize(hCommandQueue, UINT32_MAX);
    check_error(errno, "zeCommandQueueSynchronize");

    gettimeofday(&end_custom, 0);

    double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
        + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
    total_custom_time += duration_custom;

    double avg_custom_all = total_custom_time / NUM_ROUNDS;

    printf("  Time to run custom match = %f ms\n\n", avg_custom_all);


    // CLEANING
    errno = zeKernelDestroy(hKernelAllTrue);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelAllFalse);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelAnySimple);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelAnyAlternate);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelAnyUnique);
    check_error(errno, "zeKernelDestroy");

    errno = zeModuleDestroy(hModuleAllTrue);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleAllFalse);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleAnySimple);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleAnyAlternate);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleAnyUnique);
    check_error(errno, "zeModuleDestroy");

    errno =  zeCommandListDestroy(hCommandList);
    check_error(errno, "zeCommandListDestroy");

    errno = zeCommandQueueDestroy(hCommandQueue);
    check_error(errno, "zeCommandQueueDestroy");
    return 0;
}