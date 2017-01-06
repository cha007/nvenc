/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This example demonstrates how to use the Video Decode Library with CUDA
 * bindings to interop between NVDECODE(using CUDA surfaces) and DX9 textures.  
 * Post-Process video (de-interlacing) is suported with this sample.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#endif

// CUDA Header includes
#include "dynlink_nvcuvid.h"  // <nvcuvid.h>
#include "dynlink_cuda.h"     // <cuda.h>
#include "dynlink_cudaD3D11.h" // <cudaD3D11.h>
#include "dynlink_builtin_types.h"	  // <builtin_types.h>

// CUDA utilities and system includes
#include "helper_functions.h"
#include "helper_cuda_drvapi.h"

// cudaDecodeD3D11 related helper functions
#include "FrameQueue.h"
#include "VideoSource.h"
#include "VideoParser.h"
#include "VideoDecoder.h"
#include "ImageDX.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"

// Include files
#include <math.h>
#include <memory>
#include <iostream>
#include <cassert>

const char *sAppName     = "NVDECODE/D3D11 Video Decoder";
const char *sAppFilename = "NVDecodeD3D11";
const char *sSDKname     = "NVDecodeD3D11";

#define VIDEO_SOURCE_FILE_0 "11111.mp4"
#define VIDEO_SOURCE_FILE_1 "22222.mp4"

StopWatchInterface *frame_timer  = NULL;
StopWatchInterface *global_timer = NULL;

int                 g_DeviceID    = 0;
bool                g_bDone       = false;
bool                g_bRunning    = false;
bool                g_bAutoQuit   = false;
bool                g_bUseVsync   = true;
bool                g_bFirstFrame = true;
bool                g_bLoop       = false;
bool                g_bUpdateCSC  = true;
bool                g_bIsProgressive = true; // assume it is progressive, unless otherwise noted
bool                g_bException  = false;
bool                g_bWaived     = false;

HWND                g_hWnd = NULL;
WNDCLASSEX          *g_wc = NULL;

CUvideoctxlock       g_CtxLock = NULL;

float present_fps, decoded_fps, total_time = 0.0f;

FrameQueue    *g_pFrameQueue_0 = 0;
VideoSource   *g_pVideoSource_0 = 0;
VideoParser   *g_pVideoParser_0 = 0;
VideoDecoder  *g_pVideoDecoder_0 = 0;

FrameQueue    *g_pFrameQueue_1 = 0;
VideoSource   *g_pVideoSource_1 = 0;
VideoParser   *g_pVideoParser_1 = 0;
VideoDecoder  *g_pVideoDecoder_1 = 0;

ID3D11Device  *g_pD3DDevice;
ID3D11DeviceContext *g_pContext;
IDXGISwapChain *g_pSwapChain;

// These are CUDA function pointers to the CUDA kernels
CUmoduleManager   *g_pCudaModule;

CUmodule           cuModNV12toARGB       = 0;
CUfunction         g_kernelNV12toARGB    = 0;
CUfunction         g_kernelPassThru      = 0;

CUcontext          g_oContext = 0;
CUdevice           g_oDevice  = 0;

eColorSpace        g_eColorSpace = ITU601;
float              g_nHue        = 0.0f;

ImageDX       *g_pImageDX		= 0;
CUdeviceptr    g_pRgba = 0;
CUarray        g_backBufferArray = 0;

CUVIDEOFORMAT g_stFormat;

unsigned int g_nWindowWidth  = 0;
unsigned int g_nWindowHeight = 0;

unsigned int g_nVideoWidth  = 0;
unsigned int g_nVideoHeight = 0;

unsigned int g_FrameCount = 0;
unsigned int g_DecodeFrameCount = 0;
unsigned int g_fpsCount = 0;      // FPS count for averaging
unsigned int g_fpsLimit = 16;     // FPS limit for sampling timer;
CUdeviceptr    g_pInteropFrame_0[3] = { 0, 0, 0 }; // if we're using CUDA malloc
CUdeviceptr    g_pInteropFrame_1[3] = { 0, 0, 0 }; // if we're using CUDA malloc
CUdeviceptr    g_pInteropFrame_2[3] = { 0, 0, 0 }; // if we're using CUDA malloc
// Forward declarations
bool    initD3D11(HWND hWnd, int *pbTCC);
void	shutdown();

HRESULT initD3D11Surface(unsigned int nWidth, unsigned int nHeight);
HRESULT freeDestSurface();

bool loadVideoSource(unsigned int &width, unsigned int &height,
                     unsigned int &dispWidth, unsigned int &dispHeight);

HRESULT initCudaResources(int bTCC);
void freeCudaResources(bool bDestroyContext);

bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive, 
	FrameQueue    *pFrameQueue,
	VideoSource   *pVideoSource,
	VideoParser   *pVideoParser,
	VideoDecoder  *pVideoDecoder,
	int offset);
void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
CUdeviceptr *ppTextureData, size_t nTexturePitch,
CUmodule cuModNV12toARGB,
CUfunction fpCudaKernel, CUstream streamID, int width, int height);

void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
CUarray array,
CUmodule cuModNV12toARGB,
CUfunction fpCudaKernel, CUstream streamID, int nWidth, int nHeight);

HRESULT drawScene(int field_num);
void renderVideoFrame(HWND hWnd);

HRESULT cleanup(bool bDestroyContext);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#endif

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

void printStatistics()
{
    int   hh, mm, ss, msec;

    present_fps = 1.f / (total_time / (g_FrameCount * 1000.f));
    decoded_fps = 1.f / (total_time / (g_DecodeFrameCount * 1000.f));

    msec = ((int)total_time % 1000);
    ss   = (int)(total_time/1000) % 60;
    mm   = (int)(total_time/(1000*60)) % 60;
    hh   = (int)(total_time/(1000*60*60)) % 60;

    printf("\n[%s] statistics\n", sSDKname);
    printf("\t Video Length (hh:mm:ss.msec)   = %02d:%02d:%02d.%03d\n", hh, mm, ss, msec);

    printf("\t Frames Presented (inc repeats) = %d\n", g_FrameCount);
    printf("\t Average Present Rate     (fps) = %4.2f\n", present_fps);

    printf("\t Frames Decoded   (hardware)    = %d\n", g_DecodeFrameCount);
    printf("\t Average Rate of Decoding (fps) = %4.2f\n", decoded_fps);
}

void computeFPS(HWND hWnd)
{
    sdkStopTimer(&frame_timer);

    if (g_bRunning)
    {
        g_fpsCount++;

        if (!(g_pFrameQueue_0->isEndOfDecode() && g_pFrameQueue_0->isEmpty()))
        {
            g_FrameCount++;
        }
    }

    char sFPS[256];
    std::string sDecodeStatus;

    if (g_pFrameQueue_0->isEndOfDecode() && g_pFrameQueue_0->isEmpty())
    {
        sDecodeStatus = "STOP (End of File)\0";

        // we only want to record this once
        if (total_time == 0.0f)
        {
            total_time = sdkGetTimerValue(&global_timer);
        }

        sdkStopTimer(&global_timer);

        if (g_bAutoQuit)
        {
            g_bRunning = false;
            g_bDone    = true;
        }
    }
    else
    {
        if (!g_bRunning)
        {
            sDecodeStatus = "PAUSE\0";
            sprintf(sFPS, "%s [%s] - [%s %d] / Vsync %s",
                    sAppName, sDecodeStatus.c_str(),
                    (g_bIsProgressive ? "Frame" : "Field"), g_DecodeFrameCount,
                    g_bUseVsync   ? "ON" : "OFF");

			SetWindowText(hWnd, sFPS);
			UpdateWindow(hWnd);            
        }
        else
        {			
			sDecodeStatus = "PLAY\0";            
        }

        if (g_fpsCount == g_fpsLimit)
        {
            float ifps = 1.f / (sdkGetAverageTimerValue(&frame_timer) / 1000.f);

            sprintf(sFPS, "[%s] [%s] - [%3.1f fps, %s %d] / Vsync %s",
                    sAppName, sDecodeStatus.c_str(), ifps,
                    (g_bIsProgressive ? "Frame" : "Field"), g_DecodeFrameCount,
                    g_bUseVsync   ? "ON" : "OFF");

			SetWindowText(hWnd, sFPS);
			UpdateWindow(hWnd);            

            printf("[%s] - [%s: %04d, %04.1f fps, time: %04.2f (ms) ]\n",
                   sSDKname, (g_bIsProgressive ? "Frame" : "Field"), g_FrameCount, ifps, 1000.f/ifps);

            sdkResetTimer(&frame_timer);
            g_fpsCount = 0;
        }
    }

    sdkStartTimer(&frame_timer);
}

bool loadVideoSource(
	unsigned int &width, unsigned int &height,
	unsigned int &dispWidth, unsigned int &dispHeight)
{
	// Now verify the input video file is legit
	FILE *fp = NULL;

	FOPEN(fp, VIDEO_SOURCE_FILE_0, "r");
	g_pFrameQueue_0 = new FrameQueue;
	g_pVideoSource_0 = new VideoSource(VIDEO_SOURCE_FILE_0, g_pFrameQueue_0);

	FOPEN(fp, VIDEO_SOURCE_FILE_1, "r");
	g_pFrameQueue_1 = new FrameQueue;
	g_pVideoSource_1 = new VideoSource(VIDEO_SOURCE_FILE_1, g_pFrameQueue_1);

	// retrieve the video source (width,height)
	g_pVideoSource_0->getDisplayDimensions(width, height);
	g_pVideoSource_0->getDisplayDimensions(dispWidth, dispHeight);

	memset(&g_stFormat, 0, sizeof(CUVIDEOFORMAT));
	std::cout << (g_stFormat = g_pVideoSource_0->format()) << std::endl;

	bool IsProgressive = 0;
	g_pVideoSource_0->getProgressive(IsProgressive);
	return IsProgressive;
}

HRESULT initCudaResources(int bTCC)
{
	HRESULT hr = S_OK;
	CUdevice cuda_device;

	cuda_device = gpuGetMaxGflopsDeviceIdDRV();
	checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));

	// get compute capabilities and the devicename
	int major, minor;
	size_t totalGlobalMem;
	char deviceName[256];
	checkCudaErrors(cuDeviceComputeCapability(&major, &minor, g_oDevice));
	checkCudaErrors(cuDeviceGetName(deviceName, 256, g_oDevice));
	printf("> Using GPU Device %d: %s has SM %d.%d compute capability\n", cuda_device, deviceName, major, minor);

	checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, g_oDevice));
	printf("  Total amount of global memory:     %4.4f MB\n", (float)totalGlobalMem / (1024 * 1024));

	// Create CUDA Device w/ D3D11 interop (if WDDM), otherwise CUDA w/o interop (if TCC)
	// (use CU_CTX_BLOCKING_SYNC for better CPU synchronization)	
	checkCudaErrors(cuD3D11CtxCreate(&g_oContext, &g_oDevice, CU_CTX_BLOCKING_SYNC, g_pD3DDevice));

	try
	{
		// Initialize CUDA releated Driver API (32-bit or 64-bit), depending the platform running
		if (sizeof(void *) == 4)
		{
			g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_Win32.ptx", ".", 2, 2, 2);
		}
		else
		{
			g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_x64.ptx", ".", 2, 2, 2);
		}
	}
	catch (char const *p_file)
	{
		// If the CUmoduleManager constructor fails to load the PTX file, it will throw an exception
		printf("\n>> CUmoduleManager::Exception!  %s not found!\n", p_file);
		printf(">> Please rebuild NV12ToARGB_drvapi.cu or re-install this sample.\n");
		return E_FAIL;
	}

	g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi", &g_kernelNV12toARGB);
	g_pCudaModule->GetCudaFunction("Passthru_drvapi", &g_kernelPassThru);

	/////////////////Change///////////////////////////
	// Now we create the CUDA resources and the CUDA decoder context
	// bind the context lock to the CUDA context
	CUresult result = cuvidCtxLockCreate(&g_CtxLock, g_oContext);
	CUVIDEOFORMATEX oFormatEx;
	memset(&oFormatEx, 0, sizeof(CUVIDEOFORMATEX));
	oFormatEx.format = g_stFormat;

	if (result != CUDA_SUCCESS)
	{
		printf("cuvidCtxLockCreate failed: %d\n", result);
		assert(0);
	}

	g_pVideoDecoder_0 = new VideoDecoder(g_pVideoSource_0->format(), g_oContext, cudaVideoCreate_PreferCUVID, g_CtxLock);
	g_pVideoParser_0 = new VideoParser(g_pVideoDecoder_0, g_pFrameQueue_0, &oFormatEx);
	g_pVideoSource_0->setParser(*g_pVideoParser_0);

	g_pVideoDecoder_1 = new VideoDecoder(g_pVideoSource_1->format(), g_oContext, cudaVideoCreate_PreferCUVID, g_CtxLock);
	g_pVideoParser_1 = new VideoParser(g_pVideoDecoder_1, g_pFrameQueue_1, &oFormatEx);
	g_pVideoSource_1->setParser(*g_pVideoParser_1);

	initD3D11Surface(g_pVideoDecoder_0->targetWidth(),	g_pVideoDecoder_0->targetHeight());
	checkCudaErrors(cuMemAlloc(&g_pRgba, g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));

	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_0[0], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));
	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_0[1], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));
	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_0[2], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));

	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_1[0], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));
	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_1[1], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));
	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_1[2], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));

	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_2[0], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));
	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_2[1], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));
	checkCudaErrors(cuMemAlloc(&g_pInteropFrame_2[2], g_pVideoDecoder_0->targetWidth() * g_pVideoDecoder_0->targetHeight() * 4));
	CUcontext cuCurrent = NULL;
	result = cuCtxPopCurrent(&cuCurrent);
	cuCtxPushCurrent(g_oContext);//important!!!!!!!!
	if (result != CUDA_SUCCESS)
	{
		printf("cuCtxPopCurrent: %d\n", result);
		assert(0);
	}

	/////////////////////////////////////////
	return ((g_pCudaModule && g_pVideoDecoder_0 && g_pImageDX) ? S_OK : E_FAIL);
}

void
freeCudaResources(bool bDestroyContext)
{
	if (g_pVideoParser_0)
	{
		delete g_pVideoParser_0;
	}

	if (g_pVideoDecoder_0)
	{
		delete g_pVideoDecoder_0;
	}

	if (g_pVideoSource_0)
	{
		delete g_pVideoSource_0;
	}

	if (g_pFrameQueue_0)
	{
		delete g_pFrameQueue_0;
	}

	if (g_CtxLock)
	{
		checkCudaErrors(cuvidCtxLockDestroy(g_CtxLock));
	}

	if (g_oContext && bDestroyContext)
	{
		checkCudaErrors(cuCtxDestroy(g_oContext));
		g_oContext = NULL;
	}
}

// Run the Cuda part of the computation (if g_pFrameQueue is empty, then return false)
bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive, 
	FrameQueue    *FrameQueue,	
	VideoSource   *VideoSource, 	
	VideoParser   *VideoParser, 
	VideoDecoder  *VideoDecoder, int offset)
{
	CUVIDPARSERDISPINFO oDisplayInfo;
	bool isDequeueOK = FrameQueue->dequeue(&oDisplayInfo);
	if (isDequeueOK)
	{
		CCtxAutoLock lck(g_CtxLock);
		// Push the current CUDA context (only if we are using CUDA decoding path)
		CUresult result = cuCtxPushCurrent(g_oContext);

		CUdeviceptr  pDecodedFrame[3] = { 0, 0, 0 };

		*pbIsProgressive = oDisplayInfo.progressive_frame;
		g_bIsProgressive = oDisplayInfo.progressive_frame ? true : false;

		int num_fields = 1;
		if (g_bUseVsync) {
			num_fields = std::min(2 + oDisplayInfo.repeat_first_field, 3);
		}
		nRepeats = num_fields;

		CUVIDPROCPARAMS oVideoProcessingParameters;
		memset(&oVideoProcessingParameters, 0, sizeof(CUVIDPROCPARAMS));

		oVideoProcessingParameters.progressive_frame = oDisplayInfo.progressive_frame;
		oVideoProcessingParameters.top_field_first = oDisplayInfo.top_field_first;
		oVideoProcessingParameters.unpaired_field = (oDisplayInfo.repeat_first_field < 0);

		for (int active_field = 0; active_field < num_fields; active_field++)
		{
			unsigned int nDecodedPitch = 0;
			unsigned int nWidth = 0;
			unsigned int nHeight = 0;

			oVideoProcessingParameters.second_field = active_field;

			// map decoded video frame to CUDA surfae
			VideoDecoder->mapFrame(oDisplayInfo.picture_index, &pDecodedFrame[active_field], &nDecodedPitch, &oVideoProcessingParameters);
			nWidth = VideoDecoder->targetWidth();
			nHeight = VideoDecoder->targetHeight();
			// map DirectX texture to CUDA surface
			size_t nTexturePitch = 0;

			printf("%d %s = %02d, PicIndex = %02d, OutputPTS = %08d\n",offset,
				(oDisplayInfo.progressive_frame ? "Frame" : "Field"),
				g_DecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);

					
			// Upload the Color Space Conversion Matrices
			if (g_bUpdateCSC)
			{
				// CCIR 601/709
				float hueColorSpaceMat[9];
				setColorSpaceMatrix(g_eColorSpace, hueColorSpaceMat, g_nHue);
				updateConstantMemory_drvapi(g_pCudaModule->getModule(), hueColorSpaceMat);
				g_bUpdateCSC = false;
			}
				
			// Final Stage: NV12toARGB color space conversion
			CUresult eResult;
			if (offset != 3){
				eResult = cudaLaunchNV12toARGBDrv(pDecodedFrame[active_field], nDecodedPitch,
					g_pInteropFrame_0[active_field], nWidth * 4,
					nWidth, nHeight, g_kernelNV12toARGB, 0);
			}
			else{
				eResult = cudaLaunchNV12toARGBDrv(pDecodedFrame[active_field], nDecodedPitch,
					g_pInteropFrame_1[active_field], nWidth * 4,
					nWidth, nHeight, g_kernelNV12toARGB, 0);
			}
			
			VideoDecoder->unmapFrame(pDecodedFrame[active_field]);
			g_DecodeFrameCount++;
		}
		// Detach from the Current thread
		checkCudaErrors(cuCtxPopCurrent(NULL));
		// release the frame, so it can be re-used in decoder
		FrameQueue->releaseFrame(&oDisplayInfo);
	}
	else
	{
		// Frame Queue has no frames, we don't compute FPS until we start
		return false;
	}

	// check if decoding has come to an end.
	// if yes, signal the app to shut down.
	if (!VideoSource->isStarted() && FrameQueue->isEndOfDecode() && FrameQueue->isEmpty())
	{
		// Let's just stop, and allow the user to quit, so they can at least see the results
		VideoSource->stop();

		// If we want to loop reload the video file and restart
// 		if (g_bLoop && !g_bAutoQuit)
// 		{
// 			reinitCudaResources();
// 			g_FrameCount = 0;
// 			g_DecodeFrameCount = 0;
// 			g_pVideoSource->start();
// 		}

		if (g_bAutoQuit)
		{
			g_bDone = true;
		}
	}

// 	if (isDequeueOK){
// 		for (int active_field = 0; active_field < nRepeats; active_field++)
// 		{
// 			unsigned int nWidth = 0;
// 			unsigned int nHeight = 0;
// 			nWidth = g_pVideoDecoder_0->targetWidth();
// 			nHeight = g_pVideoDecoder_0->targetHeight();
// 
// 			g_backBufferArray = 0;
// 			// map the texture surface
// 			g_pImageDX->map(&g_backBufferArray, active_field, 0);
// 			CUDA_MEMCPY2D memcpy2D = { 0 };
// 			memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
// 			memcpy2D.srcDevice = g_pInteropFrame[active_field];
// 			memcpy2D.srcPitch = nWidth * 4;
// 			memcpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
// 			memcpy2D.dstArray = g_backBufferArray;
// 			memcpy2D.dstPitch = nWidth * 4;
// 			memcpy2D.WidthInBytes = nWidth * 4;
// 			memcpy2D.Height = nHeight;
// 
// 			// clear the surface to solid white
// 			checkCudaErrors(cuMemcpy2D(&memcpy2D));
// 			// unmap the texture surface
// 			g_pImageDX->unmap(active_field, 0);
// 		}
// 	}

	return true;
}
// This is the CUDA stage for Video Post Processing.  Last stage takes care of the NV12 to ARGB
void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
CUarray array,
CUmodule cuModNV12toARGB,
CUfunction fpCudaKernel, CUstream streamID, int nWidth, int nHeight)
{
	// Upload the Color Space Conversion Matrices
	if (g_bUpdateCSC)
	{
		// CCIR 601/709
		float hueColorSpaceMat[9];
		setColorSpaceMatrix(g_eColorSpace, hueColorSpaceMat, g_nHue);
		updateConstantMemory_drvapi(cuModNV12toARGB, hueColorSpaceMat);
		
		g_bUpdateCSC = false;	
	}

	// TODO: Stage for handling video post processing

	// Final Stage: NV12toARGB color space conversion
	CUresult eResult;
	eResult = cudaLaunchNV12toARGBDrv(*ppDecodedFrame, nDecodedPitch,
		g_pRgba, nWidth * 4,
		nWidth, nHeight, fpCudaKernel, streamID);

	CUDA_MEMCPY2D memcpy2D = { 0 };
	memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	memcpy2D.srcDevice = g_pRgba;
	memcpy2D.srcPitch = nWidth * 4;
	memcpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	memcpy2D.dstArray = array;
	memcpy2D.dstPitch = nWidth * 4;
	memcpy2D.WidthInBytes = nWidth * 4;
	memcpy2D.Height = nHeight;

	// clear the surface to solid white
	checkCudaErrors(cuMemcpy2D(&memcpy2D));
}
// This is the CUDA stage for Video Post Processing.  Last stage takes care of the NV12 to ARGB
void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch,
CUdeviceptr *ppTextureData, size_t nTexturePitch,
CUmodule cuModNV12toARGB,
CUfunction fpCudaKernel, CUstream streamID, int nWidth, int nHeight)
{
	// Upload the Color Space Conversion Matrices
	if (g_bUpdateCSC)
	{
		// CCIR 601/709
		float hueColorSpaceMat[9];
		setColorSpaceMatrix(g_eColorSpace, hueColorSpaceMat, g_nHue);
		updateConstantMemory_drvapi(cuModNV12toARGB, hueColorSpaceMat);
		
		g_bUpdateCSC = false;		
	}

	// TODO: Stage for handling video post processing

	// Final Stage: NV12toARGB color space conversion
	CUresult eResult;
	eResult = cudaLaunchNV12toARGBDrv(*ppDecodedFrame, nDecodedPitch,
		*ppTextureData, nTexturePitch,
		nWidth, nHeight, fpCudaKernel, streamID);
}

void shutdown()
{
	// clean up CUDA and OpenGL resources
	cleanup(g_bWaived ? false : true);

	{
		// Unregister windows class
		UnregisterClass(g_wc->lpszClassName, g_wc->hInstance);
	}

	if (g_bAutoQuit)
	{
		PostQuitMessage(0);
	}

	if (g_hWnd)
	{
		DestroyWindow(g_hWnd);
	}

	if (g_bWaived)
	{
		exit(EXIT_WAIVED);
	}
	else
	{
		exit(g_bException ? EXIT_FAILURE : EXIT_SUCCESS);
	}
}

inline std::string wcs2mbstring(const wchar_t *wcs)
{
	size_t len = wcslen(wcs) + 1;
	char *mbs = new char[len];
	wcstombs(mbs, wcs, len);

	std::string mbstring(mbs);
	delete mbs;
	return mbstring;
}

// Initialize Direct3D
bool
initD3D11(HWND hWnd, int *pbTCC)
{
    int dev, device_count = 0;
    char device_name[256];

    // Check for a min spec of Compute 1.1 capability before running
    checkCudaErrors(cuDeviceGetCount(&device_count));

    if ((g_DeviceID > (device_count-1)) || (g_DeviceID < 0))
    {
        printf(" >>> Invalid GPU Device ID=%d specified, only %d GPU device(s) are available.<<<\n", g_DeviceID, device_count);
        printf(" >>> Valid GPU ID (n) range is between [%d,%d]...  Exiting... <<<\n", 0, device_count-1);
        return false;
    }

    // We are specifying a GPU device, check to see if it is TCC or not
    checkCudaErrors(cuDeviceGet(&dev, g_DeviceID));
    checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

    checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
    printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");

    if (*pbTCC){
		assert(0);
    }

    HRESULT eResult = S_OK;

    bool bDeviceFound = false;
    int device;

    // Find the first CUDA capable device
    CUresult cuStatus;
	IDXGIAdapter *pAdapter = NULL;
	IDXGIFactory1 *pFactory = NULL;
	CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&pFactory);
    for (unsigned int g_iAdapter = 0; pFactory->EnumAdapters(g_iAdapter, &pAdapter) == S_OK; g_iAdapter++)
    {
		DXGI_ADAPTER_DESC desc;
		pAdapter->GetDesc(&desc);

        cuStatus = cuD3D11GetDevice(&device, pAdapter);
        printf("> Display Device: \"%s\" %s Direct3D11\n",
                wcs2mbstring(desc.Description).c_str(),
                (cuStatus == cudaSuccess) ? "supports" : "does not support");

        if (cudaSuccess == cuStatus)
        {
            bDeviceFound = true;
            break;
        }
    }
	pFactory->Release();

    // we check to make sure we have found a cuda-compatible D3D device to work on
    if (!bDeviceFound)
    {
        printf("\n");
        printf("  No CUDA-compatible Direct3D9 device available\n");
        // destroy the D3D device
        return false;
    }

    // Create the D3D Display Device
	/* Initialize D3D */
	DXGI_SWAP_CHAIN_DESC sc = { 0 };
	sc.BufferCount = 1;
	sc.BufferDesc.Width = g_nVideoWidth;
	sc.BufferDesc.Height = g_nVideoHeight;
	sc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	sc.BufferDesc.RefreshRate.Numerator = 0;
	sc.BufferDesc.RefreshRate.Denominator = 1;
	sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sc.OutputWindow = hWnd;
	sc.SampleDesc.Count = 1;
	sc.SampleDesc.Quality = 0;
	sc.Windowed = TRUE;

	HRESULT hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE,
		NULL, 0, NULL, 0, D3D11_SDK_VERSION, &sc, &g_pSwapChain, &g_pD3DDevice, NULL, &g_pContext);
	if (FAILED(hr)) {
		printf("Unable to create DX11 device and swapchain, hr=0x%x", hr);
		return false;
	}
	
    return (eResult == S_OK);
}

// Initialize Direct3D Textures (allocation and initialization)
HRESULT
initD3D11Surface(unsigned int nWidth, unsigned int nHeight)
{
    g_pImageDX = new ImageDX(g_pD3DDevice, g_pContext, g_pSwapChain,
                             nWidth, nHeight,
                             nWidth, nHeight,
                             g_bUseVsync,
                             ImageDX::BGRA_PIXEL_FORMAT); // ImageDX::LUMINANCE_PIXEL_FORMAT
    g_pImageDX->clear(0x80);

    g_pImageDX->setCUDAcontext(g_oContext);
    g_pImageDX->setCUDAdevice(g_oDevice);
    return S_OK;
}

HRESULT
freeDestSurface()
{
    if (g_pImageDX){
        delete g_pImageDX;
        g_pImageDX = NULL;
    }

    return S_OK;
}

// Draw the final result on the screen
HRESULT drawScene(int field_num)
{
    HRESULT hr = S_OK;

    // render image
	g_pImageDX->render(field_num);
    hr = g_pSwapChain->Present(g_bUseVsync ? DXGI_SWAP_EFFECT_SEQUENTIAL : DXGI_SWAP_EFFECT_DISCARD, 0);

    return S_OK;
}
bool isDequeued_0 = false;
bool isDequeued_1 = false;
// Launches the CUDA kernels to fill in the texture data
void renderVideoFrame(HWND hWnd)
{
    static unsigned int nRepeatFrame = 0;
    int bIsProgressive = 1, bFPSComputed = 0;
    bool bFramesDecoded = false;

	if (isDequeued_0 == false){
		if (0 != g_pFrameQueue_0)
		{
			// if not running, we simply don't copy new frames from the decoder
			if (g_bRunning)
			{
				bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, true, &bIsProgressive, g_pFrameQueue_0, g_pVideoSource_0, g_pVideoParser_0, g_pVideoDecoder_0, 0);
				isDequeued_0 = true;
			}
		}
		else
		{
			return;
		}
	}
	if (isDequeued_1 == false){
		if (0 != g_pFrameQueue_1)
		{
			// if not running, we simply don't copy new frames from the decoder
			if (g_bRunning)
			{
				bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, true, &bIsProgressive, g_pFrameQueue_1, g_pVideoSource_1, g_pVideoParser_1, g_pVideoDecoder_1, 3);
				isDequeued_1 = true;
			}
		}
		else
		{
			return;
		}
	}
	nRepeatFrame = 2;
    //if (bFramesDecoded){   
	if (isDequeued_0 == true && isDequeued_1 == true){
		/*
		// do your own cuda process
		for (int i = 0; i < nRepeatFrame; i++) {
			combine((unsigned int*)g_pInteropFrame_0[i], 
				(unsigned int*)g_pInteropFrame_1[i], 
				(unsigned int*)g_pInteropFrame_2[i], 
				4096, 
				2048);
		}*/

		for (int i = 0; i < nRepeatFrame; i++) {
			unsigned int nWidth = g_pVideoDecoder_0->targetWidth();
			unsigned int nHeight = g_pVideoDecoder_0->targetHeight();

			g_backBufferArray = 0;
			// map the texture surface
			g_pImageDX->map(&g_backBufferArray, i);
	
			CUDA_MEMCPY2D memcpy2D = { 0 };
			memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
			memcpy2D.srcDevice = g_pInteropFrame_0[i];
			memcpy2D.srcPitch = nWidth * 4;
			memcpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			memcpy2D.dstArray = g_backBufferArray;
			memcpy2D.dstPitch = nWidth * 4;
			memcpy2D.WidthInBytes = nWidth * 4;
			memcpy2D.Height = nHeight;

			// clear the surface to solid white
			checkCudaErrors(cuMemcpy2D(&memcpy2D));
			// unmap the texture surface
			g_pImageDX->unmap(i);

			drawScene(i);
			computeFPS(hWnd);
		}

		bFPSComputed = 1;        
		isDequeued_0 = false;
		isDequeued_1 = false;
        // Pass the Windows handle to show Frame Rate on the window title
        if (!bFPSComputed)
        {
            computeFPS(hWnd);
        }
    }
}

int main()
{
	sdkCreateTimer(&frame_timer);
	sdkResetTimer(&frame_timer);

	sdkCreateTimer(&global_timer);
	sdkResetTimer(&global_timer);
	
	//g_bLoop = true;
	if (g_bLoop == false){
		g_bAutoQuit = true;
	}

	// Initialize the CUDA and NVDECODE
	typedef HMODULE CUDADRIVER;
	CUDADRIVER hHandleDriver = 0;
	CUresult cuResult;
	cuResult = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
	cuResult = cuvidInit(0);

	// Find out the video size
	g_bIsProgressive = loadVideoSource(	g_nVideoWidth, g_nVideoHeight,
									g_nWindowWidth, g_nWindowHeight);

	// create window (after we know the size of the input file size)
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
		GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
		sAppName, NULL
	};
	RegisterClassEx(&wc);
	g_wc = &wc;

	// figure out the window size we must create to get a *client* area
	// that is of the size requested by m_dimensions.
	RECT adjustedWindowSize;
	DWORD dwWindowStyle;

	dwWindowStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
	SetRect(&adjustedWindowSize, 0, 0, g_nVideoWidth, g_nVideoHeight);
	AdjustWindowRect(&adjustedWindowSize, dwWindowStyle, false);

	g_nWindowWidth = adjustedWindowSize.right - adjustedWindowSize.left;
	g_nWindowHeight = adjustedWindowSize.bottom - adjustedWindowSize.top;

	// Create the application's window
	g_hWnd = CreateWindow(wc.lpszClassName, sAppName,
		dwWindowStyle,
		0, 0,
		1920,
		960,
		NULL, NULL, wc.hInstance, NULL);

	int bTCC = 0;
	// Initialize Direct3D
	if (initD3D11(g_hWnd, &bTCC) == false)
	{
		g_bAutoQuit = true;
		g_bWaived = true;
		shutdown();
	}

	// If we are using TCC driver, then graphics interop must be disabled
	if (bTCC)
	{
		assert(0);
	}

	// Initialize CUDA/D3D11 context and other video memory resources
	if (initCudaResources(bTCC) == E_FAIL)
	{
		g_bAutoQuit = true;
		g_bException = true;
		g_bWaived = true;
		shutdown();
	}

	g_pVideoSource_0->start();
	g_pVideoSource_1->start();
	g_bRunning = true;

	ShowWindow(g_hWnd, SW_SHOWDEFAULT);
	UpdateWindow(g_hWnd);	

	// the main loop
	sdkStartTimer(&frame_timer);
	sdkStartTimer(&global_timer);
	sdkResetTimer(&global_timer);
	
	// Standard windows loop
	while (!g_bDone)
	{
		MSG msg;
		ZeroMemory(&msg, sizeof(msg));

		while (msg.message != WM_QUIT)
		{
			if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			else
			{
				renderVideoFrame(g_hWnd);
			}

			if (g_bAutoQuit && g_bDone)
			{
				break;
			}
		}
	} // while loop	

	// we only want to record this once
	if (total_time == 0.0f)
	{
		total_time = sdkGetTimerValue(&global_timer);
	}
	sdkStopTimer(&global_timer);

	g_pFrameQueue_0->endDecode();
	g_pVideoSource_0->stop();

	printStatistics();

	g_bWaived = false;
	shutdown();
}

// Release all previously initd objects
HRESULT cleanup(bool bDestroyContext)
{
	if (bDestroyContext)
	{
		// Attach the CUDA Context (so we may properly free memroy)
		checkCudaErrors(cuCtxPushCurrent(g_oContext));

		if (g_pRgba) {
			checkCudaErrors(cuMemFree(g_pRgba));
		}

		// Detach from the Current thread
		checkCudaErrors(cuCtxPopCurrent(NULL));
	}

	freeDestSurface();

	freeCudaResources(bDestroyContext);

	// destroy the D3D device
	if (g_pD3DDevice)
	{
		g_pD3DDevice->Release();
		g_pD3DDevice = NULL;
	}

	if (g_pContext) {
		g_pContext->Release();
		g_pContext = NULL;
	}

	if (g_pSwapChain) {
		g_pSwapChain->Release();
		g_pSwapChain = NULL;
	}

	return S_OK;
}

// The window's message handler
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
        case WM_KEYDOWN:
            switch (wParam)
            {
                    // use ESC to quit application
                case VK_ESCAPE:
                    {
                        g_bDone = true;
                        PostQuitMessage(0);
                        return 0;
                    }
                    break;

                    // use space to pause playback
                case VK_SPACE:
                    {
                        g_bRunning = !g_bRunning;
                    }
                    break;
            }

            break;

        case WM_DESTROY:
            g_bDone = true;
            PostQuitMessage(0);
            return 0;

        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}