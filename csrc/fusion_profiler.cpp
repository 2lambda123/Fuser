// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cxxabi.h>
#include <iomanip>
#include <fusion_profiler.h>

namespace nvfuser {

namespace {

// Copying some code from the CUPTI samples/common code
// CUPTI buffer size 8 MB
#define BUF_SIZE (8 * 1024 * 1024)
// 8-byte alignment for the buffers
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                 \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

const char* get_demangled_name(const char *pName) {
  if (pName == nullptr) {
    return "<null>";
  }
  int status = 0;
  return abi::__cxa_demangle(pName, nullptr, nullptr, &status);
}

void record_cupti_activity(CUpti_Activity *pRecord, FILE *pFileHandle) {
  CUpti_ActivityKind activityKind = pRecord->kind;

  switch (activityKind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      CUpti_ActivityKernel8 *pKARecord = (CUpti_ActivityKernel8 *)pRecord;

      KernelProfile prof;
      prof.name.assign(get_demangled_name(pKARecord->name));
      int start = prof.name.find("kernel");
      int end = prof.name.find('(');
      prof.name = prof.name.substr(start, end - start);
      prof.device = (int)pKARecord->deviceId;
      prof.stream = pKARecord->streamId;
      prof.correlation_id = pKARecord->correlationId;
      constexpr double ms_convert = 1.0 / 1000000.0;
      prof.time_ms = static_cast<double>(pKARecord->end - pKARecord->start) * ms_convert;
      prof.grid = {pKARecord->gridX, pKARecord->gridY, pKARecord->gridZ};
      prof.block = {pKARecord->blockX, pKARecord->blockY, pKARecord->blockZ};
      prof.cluster = {pKARecord->clusterX, pKARecord->clusterY, pKARecord->clusterZ};
      prof.dynamic_shared_mem = pKARecord->dynamicSharedMemory;
      prof.static_shared_mem = pKARecord->staticSharedMemory;
      prof.registers = pKARecord->registersPerThread;

      FusionProfiler::get()->recordAsyncKernelActivity(std::move(prof));

      break;
    }
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
    {
      CUpti_ActivityExternalCorrelation *pExternalCorrelationRecord = (CUpti_ActivityExternalCorrelation *)pRecord;
      FusionProfiler::get()->recordAsyncCorrIdActivity(pExternalCorrelationRecord->externalId, pExternalCorrelationRecord->correlationId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_DRIVER:
      break;
    default:
      fprintf(pFileHandle, "  <unknown>\n");
      break;
  }
}

void record_cupti_activity_buffer(
    uint8_t *pBuffer,
    size_t validBytes,
    FILE *pFileHandle,
    void *pUserData) {
  CUpti_Activity *pRecord = nullptr;
  CUptiResult status = CUPTI_SUCCESS;

  do {
    status = cuptiActivityGetNextRecord(pBuffer, validBytes, &pRecord);
    if (status == CUPTI_SUCCESS) {
      record_cupti_activity(pRecord, stdout);
    }
    else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
       break;
    }
    else {
      NVFUSER_CUPTI_SAFE_CALL(status);
    }
  } while (true);
}

void cupti_buffer_requested(
    uint8_t **ppBuffer,
    size_t *pSize,
    size_t *pMaxNumRecords)
{
    uint8_t *pBuffer = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
    NVF_ERROR(pBuffer, "CUPTI Malloced buffer Pointer is null!");

    *pSize = BUF_SIZE;
    *ppBuffer = ALIGN_BUFFER(pBuffer, ALIGN_SIZE);
    *pMaxNumRecords = 0;
}

void cupti_buffer_completed(
    CUcontext context,
    uint32_t streamId,
    uint8_t *pBuffer,
    size_t size,
    size_t validSize)
{
    if (validSize > 0) {
      record_cupti_activity_buffer(pBuffer, validSize, stdout, nullptr); 
      //FusionProfiler::kernel_profiler()->
      //    recordKernelActivity(pBuffer, validSize);
    }

    free(pBuffer);
}

const char* profiler_state2string(const ProfilerState& pstate) {
  switch (pstate) {
    case ProfilerState::Ready:
      return "Ready";
    case ProfilerState::Running:
      return "Running";
    case ProfilerState::Finished:
      return "Finished";
    case ProfilerState::Processed:
      return "Processed";
    default:
      NVF_ERROR(false, "Unexpected ProfilerState enum value!");
  }
}
 
} // annonymous

std::ostream& operator<<(std::ostream& out, const ProfilerState& pstate) {
  return out << profiler_state2string(pstate); 
}
  
CudaEventTimer::CudaEventTimer(cudaStream_t s) : 
  stream_(s),
  start_event_(),
  stop_event_(),
  time_ms_(0.0), 
  state_(ProfilerState::Ready) {
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event_));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop_event_));
}

CudaEventTimer::~CudaEventTimer() {
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event_));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop_event_));
}

void CudaEventTimer::reset() {
  time_ms_ = 0.0;
  state_ = ProfilerState::Ready;
}

void CudaEventTimer::start() {
  NVF_CHECK(state_ == ProfilerState::Ready, "ProfilerState is not Ready! ", state_);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event_, stream_));
  state_ = ProfilerState::Running;
}

void CudaEventTimer::stop() {
  NVF_CHECK(state_ == ProfilerState::Running, "ProfilerState is not Running! ", state_);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(stop_event_, stream_));
  state_ = ProfilerState::Finished;
}

double CudaEventTimer::time() {
  if (state_ == ProfilerState::Finished) {
    float tmp{0.0};
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(
        cudaEventElapsedTime(&tmp, start_event_, stop_event_));
    time_ms_ = static_cast<double>(tmp);
    state_ = ProfilerState::Processed;
  } else {
    NVF_CHECK((state_ == ProfilerState::Processed) || (state_ == ProfilerState::Ready), "ProfilerState is not Processed or Ready! ", state_);
  }
  return time_ms_;
}

ProfilerState CudaEventTimer::state() const {
  return state_;
}

void DeviceDescriptor::generate(int _device) {
  device = static_cast<int>(_device);
  name.reserve(100);
  NVFUSER_CUDA_SAFE_CALL(
      cuDeviceGetName(name.data(), 100, device));
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &bus_width,
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
      device));
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &memory_clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));

    // Peak bandwidth calculation:
    // Bus width is given in bits, so dividing by 8 converts to bytes.
    // Clock is given in kHz. 1 GB = 1e9 bytes (don't report GiB = 1024^3 bytes)
    // A factor of 2 is multiplied to account for double data rate (DDR):
    // (clock in kHz * width in bits) * (1000 Hz / kHz) * (1 GB / 8e9 bits) * 2
    // factor = 2.5e-7
  peak_bandwidth_gbs = 2.5e-7 * static_cast<double>(memory_clock) * static_cast<double>(bus_width);
}

SegmentProfiler::SegmentProfiler(uint32_t id, bool disable_cupti) :
  disable_cupti_(disable_cupti),
  device_(-1),
  segment_id_(id),
  compile_timer_(at::cuda::getCurrentCUDAStream()),
  input_bytes_(0),
  output_bytes_(0),
  kernel_profile_state_(ProfilerState::Ready) {}

void SegmentProfiler::startCompile(int device) {
  device_ = device;
  compile_timer_.start();
}

void SegmentProfiler::stopCompile() {
  compile_timer_.stop();
}

void SegmentProfiler::startKernel(int device) {
  device_ = device;
  //NVF_CHECK(segment_id_ > -1, "Segment Id is not valid! ", segment_id_);
  NVF_CHECK(kernel_profile_state_ == ProfilerState::Ready, "ProfilerState is not Ready!", kernel_profile_state_);
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(segment_id_)));
  kernel_profile_state_ = ProfilerState::Running;
}

void SegmentProfiler::stopKernel() {
  NVF_CHECK(kernel_profile_state_ == ProfilerState::Running, "ProfilerState is not Running!", kernel_profile_state_);
  uint64_t corr_id = 0;
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &corr_id));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  NVF_CHECK(corr_id == static_cast<uint64_t>(segment_id_), "Correlation Id does not match segment id! Corr Id: ", corr_id, " Segment Id: ", segment_id_);
  kernel_profile_state_ = ProfilerState::Finished;
}

void SegmentProfiler::inputBytesAccessed(int64_t bytes) {
  input_bytes_ = bytes;
}

void SegmentProfiler::outputBytesAccessed(int64_t bytes) {
  output_bytes_ = bytes;
}

uint32_t SegmentProfiler::segmentId() const {
  return segment_id_;
}

void FusionProfile::reset() {
  time_ms = 0.0;
  host_time_ms = 0.0;
  compile_time_ms = 0.0;
  kernel_time_ms = 0.0;
  
  input_bytes = 0;
  output_bytes = 0;

  effective_bandwidth_gbs = 0.0;
  percentage_peak_bandwidth = 0.0;

  kernel_profiles.clear();
}

std::array<const char*,  25> column_strs{"Fus#", "NSegs", "Time(ms)",
    "HstTm(ms)", "CmpTm(ms)", "KerTm(ms)", "EffBw(GB/s)", "%PeakBw", "Seg#",
    "S-KerName", "S-Dev", "S-Stm", "S-KerTm(ms)", "S-CmpTm(ms)",
    "S-EffBw(GB/s)", "S-%PeakBw", "S-Grid", "S-Block", "S-Cluster",
    "S-Smem[Dyn,Stat]", "S-Regs", "S-In(MB)", "S-Out(MB)", "S-DeviceName",
    "S-PeakBw(GB/s)"};

std::ostream& operator<<(std::ostream& os, const FusionProfile& fp) {
  if (fp.fusion_id == 0) {
    os << std::left <<std::setw(5) << std::get<0>(column_strs)
       << " " << std::setw(5)  << std::get<1>(column_strs)
       << " " << std::setw(8)  << std::get<2>(column_strs)
       << " " << std::setw(9) << std::get<3>(column_strs);

    if (fp.verbose) {
      os << " " << std::setw(9) << std::get<4>(column_strs)
         << " " << std::setw(9) << std::get<5>(column_strs)
         << " " << std::setw(11) << std::get<6>(column_strs)
         << " " << std::setw(9) << std::get<7>(column_strs);
    }

    os << " " << std::setw(4)  << std::get<8>(column_strs)
       << " " << std::setw(10) << std::get<9>(column_strs)
       << " " << std::setw(5)  << std::get<10>(column_strs)
       << " " << std::setw(5)  << std::get<11>(column_strs)
       << " " << std::setw(11) << std::get<12>(column_strs)
       << " " << std::setw(11) << std::get<13>(column_strs)
       << " " << std::setw(13) << std::get<14>(column_strs)
       << " " << std::setw(9)  << std::get<15>(column_strs)
       << " " << std::setw(16) << std::get<16>(column_strs)
       << " " << std::setw(16) << std::get<17>(column_strs)
       << " " << std::setw(16) << std::get<18>(column_strs)
       << " " << std::setw(16) << std::get<19>(column_strs)
       << " " << std::setw(6)  << std::get<20>(column_strs)
       << " " << std::setw(9)  << std::get<21>(column_strs)
       << " " << std::setw(9)  << std::get<22>(column_strs);

    if (fp.verbose) {
      os << " " << std::setw(20) << std::get<23>(column_strs)
         << " " << std::setw(14) << std::get<24>(column_strs);
    }

    os << std::endl;
  }

  bool first_prof = true;
  int idx = 0;
  for (auto& kp : fp.kernel_profiles) {
    if (first_prof) {
      os << std::setfill(' ') << std::right << std::fixed 
                << std::setw(5) << fp.fusion_id
         << " " << std::setw(5) << fp.kernel_profiles.size()
         << " " << std::setw(8) << std::setprecision(3) << fp.time_ms
         << " " << std::setw(9) << std::setprecision(3) << fp.host_time_ms;
      if (fp.verbose) {
        os << std::setfill(' ') << std::right << std::fixed 
           << " " << std::setw(9)  << std::setprecision(3) << fp.compile_time_ms
           << " " << std::setw(9)  << std::setprecision(3) << fp.kernel_time_ms
           << " " << std::setw(11) << std::setprecision(2) << fp.effective_bandwidth_gbs 
           << " " << std::setw(9)  << std::setprecision(2) << fp.percentage_peak_bandwidth;
      }
      first_prof = false;
    } else {
      os << std::setfill(' ') << std::right << std::fixed 
                << std::setw(5) << "-"
         << " " << std::setw(5) << "-"
         << " " << std::setw(8) << "-"
         << " " << std::setw(9) << "-";
      if (fp.verbose) {
        os << std::setfill(' ') << std::right << std::fixed 
           << " " << std::setw(9)  << "-"
           << " " << std::setw(9)  << "-"
           << " " << std::setw(11) << "-" 
           << " " << std::setw(9)  << "-";
      }

    }
    std::stringstream grid;
    grid << "[" <<std::get<0>(kp.grid) << ", " << std::get<1>(kp.grid) << ", " << std::get<2>(kp.grid) << "]";
    std::stringstream block;
    block << "[" << std::get<0>(kp.block) << ", " << std::get<1>(kp.block) << ", " << std::get<2>(kp.block) << "]";
    std::stringstream cluster;
    cluster << "[" << std::get<0>(kp.cluster) << ", " << std::get<1>(kp.cluster) << ", " << std::get<2>(kp.cluster) << "]";
    std::stringstream smem;
    smem << "[" << kp.dynamic_shared_mem << ", " << kp.static_shared_mem << "]";
    os << std::setfill(' ') << std::right << std::fixed 
       << " " << std::setw(4)  << idx
       << " " << std::setw(10) << kp.name
       << " " << std::setw(5)  << kp.device
       << " " << std::setw(5)  << kp.stream
       << " " << std::setw(11) << std::setprecision(3) << kp.time_ms
       << " " << std::setw(11) << std::setprecision(3) << kp.compile_time_ms
       << " " << std::setw(13) << std::setprecision(2) << kp.effective_bandwidth_gbs
       << " " << std::setw(9)  << std::setprecision(2) << kp.percentage_peak_bandwidth
       << " " << std::setw(16) << grid.str()
       << " " << std::setw(16) << block.str()
       << " " << std::setw(16) << cluster.str()
       << " " << std::setw(16) << smem.str()
       << " " << std::setw(6)  << kp.registers
       << " " << std::setw(9)  << std::setprecision(3) << ((double)kp.input_bytes / 1000000.0)
       << " " << std::setw(9)  << std::setprecision(3) << ((double)kp.output_bytes / 1000000.0);
    if (fp.verbose) {
       os << " " << std::setw(20) << kp.device_name
          << " " << std::setw(14) << std::setprecision(2) << kp.peak_bandwidth_gbs;
    }
    os << std::endl;
    ++idx;
  }
  return os;
}

std::mutex FusionProfiler::singleton_lock_;
FusionProfiler* FusionProfiler::singleton_ = nullptr;

FusionProfiler::FusionProfiler(bool disable_cupti) :
  disable_cupti_(disable_cupti),
  state_(ProfilerState::Ready),
  fusion_id_(-1),
  profile_(),
  fusion_timer_(at::cuda::getCurrentCUDAStream()),
  segments_(),
  device_descriptors_(),
  kernel_profiles_(),
  corrid_2_segid_() {
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityRegisterCallbacks(
          cupti_buffer_requested, cupti_buffer_completed));
}

void FusionProfiler::reset() {
  state_ = ProfilerState::Ready;
  ++fusion_id_; 

  profile_.reset();
  fusion_timer_.reset();
  segments_.clear();
  kernel_profiles_.clear();
  corrid_2_segid_.clear();
  segid_2_idx_.clear();
}

FusionProfiler* FusionProfiler::get(bool disable_cupti) {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new FusionProfiler(disable_cupti);
  } 
  return singleton_;
}

ProfilerState FusionProfiler::state() const {
  return state_;
}

void FusionProfiler::createSegments(size_t num) {
  NVF_CHECK(state_ == ProfilerState::Running, "FusionProfiler state is not Running!", state_);
  segments_.reserve(num);
  uint32_t hash_preamble = (0xffff & fusion_id_) << 15;
  for (size_t i = 0; i < num; ++i) {
    uint32_t id = hash_preamble | (0xffff & static_cast<uint32_t>(i));
    segid_2_idx_[id] = segments_.size();
    segments_.emplace_back(id, disable_cupti_);
  }
}
SegmentProfiler& FusionProfiler::segment(size_t idx) {
  return segments_.at(idx);
}

void FusionProfiler::start() {
  reset();
  fusion_timer_.start();
  state_ = ProfilerState::Running;
}

void FusionProfiler::stop() {
  NVF_CHECK(state_ == ProfilerState::Running, "FusionProfiler state is not Running!", state_);
  fusion_timer_.stop();
  state_ = ProfilerState::Finished;
  profile_.time_ms = fusion_timer_.time();
  kernel_profiles_.reserve(segments_.size());
  profile_.kernel_profiles.resize(segments_.size());
 
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityFlushAll(0));

  NVF_CHECK(kernel_profiles_.size() >= segments_.size(), "All of the kernel profiles have not been recorded!");
 
  double compile_time_ms = 0.0;
  double kernel_time_ms = 0.0;
  constexpr double mb_divider = 1.0 / 1.0e6;
  for(auto &kp : kernel_profiles_) {
    auto corr_id = kp.correlation_id;
    if (corrid_2_segid_.count(corr_id) == 0) {
      continue;
    }
    NVF_CHECK(kp.device >= 0, "Device Descriptor index is not valid! ", kp.device);
    if ((size_t)kp.device >= device_descriptors_.size()) {
      device_descriptors_.resize(kp.device + 1);
    }
    NVF_CHECK((size_t)kp.device < device_descriptors_.size(), "Device idx is beyond size of Device Descriptors! ", kp.device);
    if (device_descriptors_[kp.device].device != kp.device) {
      device_descriptors_[kp.device].generate(kp.device);
    }
    kp.device_name = device_descriptors_[kp.device].name;
    kp.peak_bandwidth_gbs = device_descriptors_[kp.device].peak_bandwidth_gbs;
    NVF_CHECK(corrid_2_segid_.count(corr_id) > 0, "Correlation Id is not found in corrid -> segid hashmap! ", corr_id);
    auto seg_id = corrid_2_segid_[corr_id];
    NVF_CHECK(segid_2_idx_.count(seg_id) > 0, "Seg id is not found in seg id -> idx hashmap! ", seg_id);
    auto kp_idx = segid_2_idx_[seg_id];
    NVF_CHECK(kp_idx < profile_.kernel_profiles.size(), "Index is out of range of Kernel Profiles size! ", kp_idx, " ", profile_.kernel_profiles.size());
    NVF_CHECK(segments_[kp_idx].state() == ProfilerState::Finished, "SegmentProfiler ProfilerState is not Finished!", segments_[kp_idx].state());
    kp.input_bytes = segments_.at(kp_idx).inputBytes();
    kp.output_bytes = segments_.at(kp_idx).outputBytes();
    kp.effective_bandwidth_gbs = (double)(kp.input_bytes + kp.output_bytes) / kp.time_ms * mb_divider;
    kp.percentage_peak_bandwidth = kp.effective_bandwidth_gbs / kp.peak_bandwidth_gbs * 100.0;
    kp.compile_time_ms = segments_.at(kp_idx).compileTime();

    compile_time_ms += kp.compile_time_ms;
    kernel_time_ms += kp.time_ms;
    profile_.kernel_profiles[kp_idx] = std::move(kp);
  }

  int device = segments_[0].device();
  for (auto &seg : segments_) {
    NVF_CHECK(seg.device() == device, "All Segment profiles must be on the same device!");
  }
  profile_.fusion_id = fusion_id_;
  profile_.host_time_ms = profile_.time_ms - compile_time_ms - kernel_time_ms;
  profile_.compile_time_ms = compile_time_ms;
  profile_.kernel_time_ms = kernel_time_ms;
  profile_.effective_bandwidth_gbs = (double)(profile_.input_bytes + profile_.output_bytes) / profile_.kernel_time_ms * mb_divider;
  profile_.percentage_peak_bandwidth = profile_.effective_bandwidth_gbs / device_descriptors_[segments_[0].device()].peak_bandwidth_gbs * 100.0;

  state_ = ProfilerState::Processed;
}
  
void FusionProfiler::inputBytesAccessed(int64_t bytes) {
  NVF_CHECK(state_ == ProfilerState::Running, "FusionProfiler state is not Running!", state_);
  profile_.input_bytes = bytes;
}

void FusionProfiler::outputBytesAccessed(int64_t bytes) {
  NVF_CHECK(state_ == ProfilerState::Running, "FusionProfiler state is not Running!", state_);
  profile_.output_bytes = bytes;
}

const FusionProfile& FusionProfiler::profile() const {
  NVF_CHECK(state_ == ProfilerState::Processed, "The FusionProfile struct data is not valid because it has not been processed! ", state_);
  return profile_;
}

void FusionProfiler::recordAsyncCorrIdActivity(uint32_t seg_id, uint32_t corr_id) {
  NVF_CHECK(corrid_2_segid_.count(corr_id) == 0, "Segment Correlation Activity asociated with this correlation id already exists! ", corr_id);
  corrid_2_segid_[corr_id] = seg_id;
}

void FusionProfiler::recordAsyncKernelActivity(KernelProfile prof) {
  kernel_profiles_.emplace_back(std::move(prof));
}

} // namespace nvfuser