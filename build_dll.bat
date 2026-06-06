@echo off
echo [*] Initializing MSVC Environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

echo [*] Compiling MecanTensor 40 DLL with Vision Engine...
cl /nologo /EHsc /O2 /arch:AVX2 /openmp /LD /Iinclude /Iinclude\mecan /Isrc ^
  src\core\allocator.cpp ^
  src\tensor.cpp ^
  src\ops\math.cpp ^
  src\ops\conv.cpp ^
  src\io\serialization.cpp ^
  src\autograd\variable.cpp ^
  src\autograd\engine.cpp ^
  src\autograd\ops_backward.cpp ^
  src\python_bridge.cpp ^
  src\optim\lgc.cpp ^
  src\ops\bitlinear.cpp ^
  src\ops\attention.cpp ^
  src\ops\spatial.cpp ^
  src\qsbits\qsbits.cpp ^
  src\midbits\midbits.cpp ^
  src\nn\linear.cpp ^
  src\nn\loss.cpp ^
  src\nn\ntm.cpp ^
  src\trainer.cpp ^
  src\vision\core_vision.cpp ^
  src\hal\registry.cpp ^
  src\runtime\pager.cpp ^
  src\runtime\backend_registry.cpp ^
  src\runtime\quant_runtime.cpp ^
  src\runtime\scheduler.cpp ^
  src\distributed\collective.cpp ^
  src\hlas\cpu_engine.cpp ^
  src\hlas\gpu_engine.cpp ^
  src\hlas\npu_engine.cpp ^
  src\hlas\hybrid_engine.cpp ^
  src\ops\pool.cpp ^
  src\ops\norm.cpp ^
  src\ops\upsample.cpp ^
  src\ops\detect_utils.cpp ^
  src\ops\pad.cpp ^
  src\vision\detect_engine.cpp ^
  src\vision\object_engine.cpp ^
  src\vision\color_engine.cpp ^
  src\vision\motion_engine.cpp ^
  src\vision\vector_engine.cpp ^
  src\vision\light_engine.cpp ^
  /Fe:mecantensor_40.dll

if exist mecantensor_40.dll (
    echo [*] DLL Compiled Successfully!

echo [*] Compiling and Executing GPU Hardware Probe...
cl /nologo /EHsc probe.cpp src\hlas\gpu_engine.cpp /I"." /link /out:probe.exe
if %ERRORLEVEL% == 0 (
    probe.exe
)
) else (
    echo [!] Compilation failed.
)
