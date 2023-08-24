#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
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
    errno = read_from_binary(&program_file_all, &program_size_all, "all_sync.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_any; size_t program_size_any;
    errno = read_from_binary(&program_file_any, &program_size_any, "any_sync.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_ballot; size_t program_size_ballot;
    errno = read_from_binary(&program_file_ballot, &program_size_ballot, "ballot_sync.ar");
    check_error(errno, "read_from_binary");

    // unsigned char* program_file_active; size_t program_size_active;
    // errno = read_from_binary(&program_file_active, &program_size_active, "activemask.ar");
    // check_error(errno, "read_from_binary");

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

    ze_module_desc_t moduleDescBallot = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_ballot,
        program_file_ballot,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleBallot;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescBallot, &hModuleBallot, NULL);

    // ze_module_desc_t moduleDescActive = {
    //     ZE_STRUCTURE_TYPE_MODULE_DESC,
    //     NULL,
    //     ZE_MODULE_FORMAT_NATIVE,
    //     program_size_active,
    //     program_file_active,
    //     NULL,
    //     NULL
    // };
    // ze_module_handle_t hModuleActive;
    // errno = zeModuleCreate(hContext, hDevice, &moduleDescActive, &hModuleActive, NULL);
    // check_error(errno, "zeModuleCreate");


    // KERNEL
    // change kernel desc var names
    // test on the remote stuff
    // verify it works
    ze_kernel_desc_t kernelDescAllPass = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_all_sync_custom_pass"
    };

    ze_kernel_desc_t kernelDescAllFail = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_all_sync_custom_fail"
    };

    ze_kernel_desc_t kernelDescAnyC2 = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_any_sync_custom_two"
    };

    ze_kernel_desc_t kernelDescAnyC4 = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_any_sync_custom_four"
    };

    // ze_kernel_desc_t kernelDescActiveC2 = {
    //     ZE_STRUCTURE_TYPE_KERNEL_DESC,
    //     NULL,
    //     0, // flags
    //     "test_activemask_custom_two"
    // };

    // ze_kernel_desc_t kernelDescActiveC4 = {
    //     ZE_STRUCTURE_TYPE_KERNEL_DESC,
    //     NULL,
    //     0, // flags
    //     "test_activemask_custom_four"
    // };

    ze_kernel_desc_t kernelDescBallotC2 = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_ballot_sync_custom_two"
    };

    ze_kernel_desc_t kernelDescBallotC4 = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_ballot_sync_custom_four"
    };

    uint32_t groupSizeX =  (uint32_t) atoi(argv[1]);
    uint32_t numGroupsX =  (uint32_t) atoi(argv[2]);
    ze_kernel_handle_t kernelHandleAllPass, kernelHandleAllFail, kernelHandleAnyC2, kernelHandleAnyC4,
        // kernelHandleActiveC2, kernelHandleActiveC4,
        kernelHandleBallotC2, kernelHandleBallotC4;

    int all_sync_shared_var_arr_p[32];
    int all_sync_updated_p[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    int all_sync_shared_var_arr_f[32];
    int all_sync_updated_f[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    int any_sync_shared_var_arr2[32];
    int any_sync_updated2[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};
    
    int any_sync_shared_var_arr4[32];
    int any_sync_updated4[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    int ballot_sync_shared_var_arr2[32];
    int ballot_sync_updated2[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};
    
    int ballot_sync_shared_var_arr4[32];
    int ballot_sync_updated4[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    errno = zeKernelCreate(hModuleAll, &kernelDescAllPass, &kernelHandleAllPass);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAll, &kernelDescAllFail, &kernelHandleAllFail);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAny, &kernelDescAnyC2, &kernelHandleAnyC2);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAny, &kernelDescAnyC4, &kernelHandleAnyC4);
    check_error(errno, "zeKernelCreate");

    // errno = zeKernelCreate(hModuleActive, &kernelDescActiveC2, &kernelHandleActiveC2);
    // check_error(errno, "zeKernelCreate");

    // errno = zeKernelCreate(hModuleActive, &kernelDescActiveC4, &kernelHandleActiveC4);
    // check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleBallot, &kernelDescBallotC2, &kernelHandleBallotC2);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleBallot, &kernelDescBallotC4, &kernelHandleBallotC4);
    check_error(errno, "zeKernelCreate");

    ze_group_count_t launchArgs = { numGroupsX, 1, 1 };
    // Append launch kernel
    zeKernelSetArgumentValue(kernelHandleAllPass, 0, sizeof(int)*32, all_sync_shared_var_arr_p);
    zeKernelSetArgumentValue(kernelHandleAllPass, 1, sizeof(int)*32, all_sync_updated_p);
    zeKernelSetGroupSize(kernelHandleAllPass, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(kernelHandleAllFail, 0, sizeof(int)*32, all_sync_shared_var_arr_f);
    zeKernelSetArgumentValue(kernelHandleAllFail, 1, sizeof(int)*32, all_sync_updated_f);
    zeKernelSetGroupSize(kernelHandleAllFail, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(kernelHandleAnyC2, 0, sizeof(int)*32, any_sync_shared_var_arr2);
    zeKernelSetArgumentValue(kernelHandleAnyC2, 1, sizeof(int)*32, any_sync_updated2);
    zeKernelSetGroupSize(kernelHandleAnyC2, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(kernelHandleAnyC4, 0, sizeof(int)*32, any_sync_shared_var_arr4);
    zeKernelSetArgumentValue(kernelHandleAnyC4, 1, sizeof(int)*32, any_sync_updated4);
    zeKernelSetGroupSize(kernelHandleAnyC4, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(kernelHandleBallotC2, 0, sizeof(int)*32, ballot_sync_shared_var_arr2);
    zeKernelSetArgumentValue(kernelHandleBallotC2, 1, sizeof(int)*32, ballot_sync_updated2);
    zeKernelSetGroupSize(kernelHandleBallotC2, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(kernelHandleBallotC4, 0, sizeof(int)*32, ballot_sync_shared_var_arr4);
    zeKernelSetArgumentValue(kernelHandleBallotC4, 1, sizeof(int)*32, ballot_sync_updated4);
    zeKernelSetGroupSize(kernelHandleBallotC4, groupSizeX, 1, 1);

    // int global_now2 = 0;
    // zeKernelSetArgumentValue(kernelHandleActiveC2, 0, sizeof(int), &global_now2);
    // zeKernelSetGroupSize(kernelHandleActiveC2, groupSizeX, 1, 1);

    // int global_now4 = 0;
    // zeKernelSetArgumentValue(kernelHandleActiveC4, 0, sizeof(int), &global_now4);
    // zeKernelSetGroupSize(kernelHandleActiveC4, groupSizeX, 1, 1);

    zeCommandListAppendLaunchKernel(hCommandList, kernelHandleAllPass, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, kernelHandleAllFail, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, kernelHandleAnyC2, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, kernelHandleAnyC4, &launchArgs, NULL, 0, NULL);
    // zeCommandListAppendLaunchKernel(hCommandList, kernelHandleActiveC2, &launchArgs, NULL, 0, NULL);
    // zeCommandListAppendLaunchKernel(hCommandList, kernelHandleActiveC4, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, kernelHandleBallotC2, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, kernelHandleBallotC4, &launchArgs, NULL, 0, NULL);

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

    printf("  Time to run custom vote = %f ms\n\n", avg_custom_all);


    // CLEANING
    errno = zeKernelDestroy(kernelHandleAllPass);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(kernelHandleAllFail);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(kernelHandleAnyC2);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(kernelHandleAnyC4);
    check_error(errno, "zeKernelDestroy");

    // errno = zeKernelDestroy(kernelHandleActiveC2);
    // check_error(errno, "zeKernelDestroy");

    // errno = zeKernelDestroy(kernelHandleActiveC4);
    // check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(kernelHandleBallotC2);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(kernelHandleBallotC4);
    check_error(errno, "zeKernelDestroy");

    errno = zeModuleDestroy(hModuleAll);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleAny);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleBallot);
    check_error(errno, "zeModuleDestroy");

    // errno = zeModuleDestroy(hModuleActive);
    // check_error(errno, "zeModuleDestroy");

    errno =  zeCommandListDestroy(hCommandList);
    check_error(errno, "zeCommandListDestroy");

    errno = zeCommandQueueDestroy(hCommandQueue);
    check_error(errno, "zeCommandQueueDestroy");
    return 0;
}