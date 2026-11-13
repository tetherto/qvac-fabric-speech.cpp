#if !__has_feature(objc_arc)
#error "parakeet-diarizer.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-diarizer.h"
#include "coreml/coreml_diarizers.h"
#include "coreml/parakeet_coreml_sidecars.h"
#include "coreml/parakeet_coreml_units.h"
#include "parakeet_coreml_path.h"
#include "parakeet_log.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char * kValidMaskName = "valid_mask";
constexpr int kTrailingDims = 2;

enum DiarizerStatus {
    kOk = 0,
    kBadArguments = 1,
    kBadSlice = 2,
    kPredictionFailed = 3,
    kMissingOutput = 4,
};

struct Binding {
    std::string name;
    const void * array = nullptr;
    int64_t rows = 0;
    int64_t columns = 0;
};

MLMultiArray * bridged(const void * array) {
    return (__bridge MLMultiArray *) array;
}

bool trailing_rows_and_columns(NSArray<NSNumber *> * shape, int64_t * rows, int64_t * columns) {
    if (shape.count == 0) return false;
    if (shape.count == 1) {
        *rows = shape[0].longLongValue;
        *columns = 1;
        return *rows > 0;
    }
    *rows = shape[shape.count - kTrailingDims].longLongValue;
    *columns = shape[shape.count - 1].longLongValue;
    return *rows > 0 && *columns > 0;
}

bool is_contiguous(MLMultiArray * array, int64_t columns) {
    NSArray<NSNumber *> * strides = array.strides;
    if (strides.count == 0 || strides[strides.count - 1].longLongValue != 1) return false;
    return strides.count == 1 || strides[strides.count - kTrailingDims].longLongValue == columns;
}

MLMultiArray * float_array(MLMultiArrayConstraint * constraint) {
    if (constraint == nil || constraint.dataType != MLMultiArrayDataTypeFloat32) return nil;
    NSError * error = nil;
    MLMultiArray * array = [[MLMultiArray alloc] initWithShape:constraint.shape
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&error];
    return error == nil ? array : nil;
}

bool bind_array(MLMultiArrayConstraint * constraint, NSString * name, Binding & binding,
                MLMultiArray ** array) {
    *array = float_array(constraint);
    if (*array == nil) return false;
    if (!trailing_rows_and_columns(constraint.shape, &binding.rows, &binding.columns)) return false;
    if (!is_contiguous(*array, binding.columns)) return false;
    binding.name = name.UTF8String;
    binding.array = CFBridgingRetain(*array);
    return true;
}

void release_bindings(std::vector<Binding> & bindings) {
    for (Binding & binding : bindings) {
        if (binding.array != nullptr) CFRelease(binding.array);
        binding.array = nullptr;
    }
    bindings.clear();
}

const parakeet_coreml_rows * find_input(const parakeet_coreml_rows * inputs, int count,
                                        const std::string & name) {
    for (int index = 0; index < count; ++index) {
        if (inputs[index].name != nullptr && name == inputs[index].name) return &inputs[index];
    }
    return nullptr;
}

const Binding * find_binding(const std::vector<Binding> & bindings, const char * name) {
    for (const Binding & binding : bindings) {
        if (name != nullptr && binding.name == name) return &binding;
    }
    return nullptr;
}

bool slice_fits(int64_t row_offset, int64_t rows, int64_t columns, const Binding & binding) {
    return row_offset >= 0 && rows >= 0 && row_offset + rows <= binding.rows &&
           columns == binding.columns;
}

bool fill_input(const Binding & binding, const parakeet_coreml_rows * slice) {
    if (slice == nullptr) return false;
    float * destination = (float *) bridged(binding.array).dataPointer;
    const int64_t total = binding.rows * binding.columns;
    if (slice->rows == 0) {
        std::fill(destination, destination + total, 0.0f);
        return true;
    }
    if (slice->data == nullptr ||
        !slice_fits(slice->row_offset, slice->rows, slice->columns, binding)) return false;
    const int64_t begin = slice->row_offset * binding.columns;
    const int64_t end = begin + slice->rows * binding.columns;
    std::fill(destination, destination + begin, 0.0f);
    std::memcpy(destination + begin, slice->data, (size_t) (end - begin) * sizeof(float));
    std::fill(destination + end, destination + total, 0.0f);
    return true;
}

void fill_mask(const Binding & mask, int64_t valid_frames) {
    float * destination = (float *) bridged(mask.array).dataPointer;
    std::fill(destination, destination + mask.rows, 0.0f);
    std::fill(destination, destination + valid_frames, 1.0f);
}

void copy_row(const float * line, int64_t column_stride, int64_t columns, float * destination) {
    for (int64_t column = 0; column < columns; ++column) {
        destination[column] = line[column * column_stride];
    }
}

bool copy_rows(MLMultiArray * array, const parakeet_coreml_rows_out & output) {
    if (array == nil || array.dataType != MLMultiArrayDataTypeFloat32) return false;
    NSArray<NSNumber *> * strides = array.strides;
    const int64_t row_stride = strides.count >= kTrailingDims
        ? strides[strides.count - kTrailingDims].longLongValue : 1;
    const int64_t column_stride = strides[strides.count - 1].longLongValue;
    const float * source = (const float *) array.dataPointer;
    for (int64_t row = 0; row < output.rows; ++row) {
        copy_row(source + (output.row_offset + row) * row_stride, column_stride,
                 output.columns, output.data + row * output.columns);
    }
    return true;
}

}  // namespace

struct parakeet_coreml_diarizer {
    const void * model = nullptr;
    const void * provider = nullptr;
    const void * options = nullptr;
    std::vector<Binding> inputs;
    std::vector<Binding> outputs;
    Binding mask;
    std::string label;
    std::mutex mutex;
};

namespace {

bool bind_inputs(parakeet_coreml_diarizer * ctx, MLModel * model,
                 NSMutableDictionary<NSString *, MLMultiArray *> * features) {
    NSDictionary<NSString *, MLFeatureDescription *> * descriptions =
        model.modelDescription.inputDescriptionsByName;
    for (NSString * name in descriptions) {
        Binding binding;
        MLMultiArray * array = nil;
        if (!bind_array(descriptions[name].multiArrayConstraint, name, binding, &array)) return false;
        features[name] = array;
        if (binding.name == kValidMaskName) {
            ctx->mask = binding;
        } else {
            ctx->inputs.push_back(binding);
        }
    }
    return ctx->mask.array != nullptr;
}

bool inputs_share_capacity(const parakeet_coreml_diarizer * ctx) {
    for (const Binding & binding : ctx->inputs) {
        if (binding.rows != ctx->mask.rows) return false;
    }
    return true;
}

bool bind_outputs(parakeet_coreml_diarizer * ctx, MLModel * model,
                  NSMutableDictionary<NSString *, id> * backings) {
    NSDictionary<NSString *, MLFeatureDescription *> * descriptions =
        model.modelDescription.outputDescriptionsByName;
    for (NSString * name in descriptions) {
        Binding binding;
        MLMultiArray * array = nil;
        if (!bind_array(descriptions[name].multiArrayConstraint, name, binding, &array)) return false;
        backings[name] = array;
        ctx->outputs.push_back(binding);
    }
    return !ctx->outputs.empty();
}

bool bind_model(parakeet_coreml_diarizer * ctx, MLModel * model) {
    NSMutableDictionary<NSString *, MLMultiArray *> * features = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, id> * backings = [NSMutableDictionary dictionary];
    if (!bind_inputs(ctx, model, features) || !inputs_share_capacity(ctx)) return false;
    if (!bind_outputs(ctx, model, backings)) return false;
    NSError * error = nil;
    MLDictionaryFeatureProvider * provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    if (provider == nil || error != nil) return false;
    MLPredictionOptions * options = [[MLPredictionOptions alloc] init];
    options.outputBackings = backings;
    ctx->provider = CFBridgingRetain(provider);
    ctx->options = CFBridgingRetain(options);
    return true;
}

bool inputs_bound(const parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows * inputs,
                  int count) {
    for (int index = 0; index < count; ++index) {
        if (find_binding(ctx->inputs, inputs[index].name) == nullptr) return false;
    }
    return true;
}

bool fill_inputs(parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows * inputs, int count) {
    for (const Binding & binding : ctx->inputs) {
        if (!fill_input(binding, find_input(inputs, count, binding.name))) return false;
    }
    return true;
}

bool outputs_fit(const parakeet_coreml_diarizer * ctx, const parakeet_coreml_rows_out * outputs,
                 int count) {
    for (int index = 0; index < count; ++index) {
        const Binding * binding = find_binding(ctx->outputs, outputs[index].name);
        if (binding == nullptr || outputs[index].data == nullptr ||
            !slice_fits(outputs[index].row_offset, outputs[index].rows, outputs[index].columns,
                        *binding)) return false;
    }
    return true;
}

bool read_outputs(id<MLFeatureProvider> result, const parakeet_coreml_rows_out * outputs,
                  int count) {
    for (int index = 0; index < count; ++index) {
        NSString * name = [NSString stringWithUTF8String:outputs[index].name];
        if (!copy_rows([result featureValueForName:name].multiArrayValue, outputs[index])) return false;
    }
    return true;
}

}  // namespace

struct parakeet_coreml_diarizer * parakeet_coreml_diarizer_init(const char * path_mlmodelc,
                                                                const char * units_env) {
    return parakeet_coreml_diarizer_init_with_units(
        path_mlmodelc, units_env != nullptr ? std::getenv(units_env) : nullptr);
}

struct parakeet_coreml_diarizer * parakeet_coreml_diarizer_init_with_units(const char * path_mlmodelc,
                                                                           const char * units) {
    if (path_mlmodelc == nullptr) return nullptr;
    @autoreleasepool {
        std::string label;
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = parakeet_coreml_units_for(
            units, /*prefer_neural_engine=*/true, &label);
        NSError * error = nil;
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path_mlmodelc]];
        MLModel * model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (model == nil || error != nil) return nullptr;
        auto * ctx = new parakeet_coreml_diarizer();
        ctx->model = CFBridgingRetain(model);
        ctx->label = label;
        if (!bind_model(ctx, model)) {
            parakeet_coreml_diarizer_free(ctx);
            return nullptr;
        }
        return ctx;
    }
}

void parakeet_coreml_diarizer_free(struct parakeet_coreml_diarizer * ctx) {
    if (ctx == nullptr) return;
    release_bindings(ctx->inputs);
    release_bindings(ctx->outputs);
    if (ctx->mask.array != nullptr) CFRelease(ctx->mask.array);
    if (ctx->options != nullptr) CFRelease(ctx->options);
    if (ctx->provider != nullptr) CFRelease(ctx->provider);
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

int64_t parakeet_coreml_diarizer_capacity(const struct parakeet_coreml_diarizer * ctx) {
    return ctx != nullptr ? ctx->mask.rows : 0;
}

int parakeet_coreml_diarizer_run(struct parakeet_coreml_diarizer * ctx,
                                 int64_t valid_frames,
                                 const struct parakeet_coreml_rows * inputs,
                                 int n_inputs,
                                 const struct parakeet_coreml_rows_out * outputs,
                                 int n_outputs) {
    if (ctx == nullptr || valid_frames <= 0 || valid_frames > ctx->mask.rows ||
        n_inputs < 0 || n_outputs <= 0 || outputs == nullptr) return kBadArguments;
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        if (!inputs_bound(ctx, inputs, n_inputs) || !fill_inputs(ctx, inputs, n_inputs) ||
            !outputs_fit(ctx, outputs, n_outputs)) {
            return kBadSlice;
        }
        fill_mask(ctx->mask, valid_frames);
        NSError * error = nil;
        MLModel * model = (__bridge MLModel *) ctx->model;
        id<MLFeatureProvider> result =
            [model predictionFromFeatures:(__bridge MLDictionaryFeatureProvider *) ctx->provider
                                  options:(__bridge MLPredictionOptions *) ctx->options
                                    error:&error];
        if (result == nil || error != nil) return kPredictionFailed;
        return read_outputs(result, outputs, n_outputs) ? kOk : kMissingOutput;
    }
}

const char * parakeet_coreml_diarizer_label(const struct parakeet_coreml_diarizer * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}

namespace parakeet {
namespace {

constexpr int kNoSidecar = -1;
constexpr char kPathSeparator = '/';
constexpr const char * kCurrentDirectory = ".";

std::string directory_of(const std::string & path) {
    const std::size_t slash = path.find_last_of(kPathSeparator);
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

bool is_directory(const std::string & path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode);
}

std::vector<int> listed_capacities(DIR * listing, const std::string & directory,
                                   const std::string & gguf_path) {
    std::vector<int> capacities;
    while (const dirent * entry = ::readdir(listing)) {
        const std::string candidate = directory + entry->d_name;
        const int capacity = coreml_diarizer_sidecar_capacity(gguf_path, candidate);
        if (capacity > 0 && is_directory(candidate)) capacities.push_back(capacity);
    }
    return capacities;
}

std::vector<int> scan_capacities(const std::string & gguf_path) {
    const std::string directory = directory_of(gguf_path);
    DIR * listing = ::opendir(directory.empty() ? kCurrentDirectory : directory.c_str());
    if (listing == nullptr) return {};
    std::vector<int> capacities = listed_capacities(listing, directory, gguf_path);
    ::closedir(listing);
    std::sort(capacities.begin(), capacities.end());
    capacities.erase(std::unique(capacities.begin(), capacities.end()), capacities.end());
    return capacities;
}

constexpr const char * kWeightsFile = "/weights/weight.bin";

uint64_t file_bytes(const std::string & path) {
    struct stat status {};
    return ::stat(path.c_str(), &status) == 0 ? static_cast<uint64_t>(status.st_size) : 0;
}

CoremlDiarizerBundle bundle_of(const std::string & gguf_path, int capacity) {
    return {capacity, file_bytes(coreml_diarizer_sidecar_path(gguf_path, capacity) + kWeightsFile)};
}

}

bool coreml_diarizers_disabled() {
    return std::getenv(kCoremlDisableEnv) != nullptr ||
           std::getenv(kCoremlDiarizerDisableEnv) != nullptr;
}

std::vector<CoremlDiarizerBundle> list_coreml_diarizer_bundles(const std::string & gguf_path) {
    const std::vector<int> capacities = scan_capacities(gguf_path);
    std::vector<CoremlDiarizerBundle> bundles(capacities.size());
    std::transform(capacities.begin(), capacities.end(), bundles.begin(),
                   [&](int capacity) { return bundle_of(gguf_path, capacity); });
    return bundles;
}

namespace {

// Runs on the runtime's load lane with the resolved units: never reads
// the environment. The capacity the bundle reports must match its file name.
// A generated bundle (request.tmpl set) loads under its cache entry's lease.
pcm::LoadResult load_diarizer(const pcm::LoadRequest & request) {
    pcm::LoadResult result;
    if (!pcm::generation::begin_load(request, result)) return result;
    const int64_t capacity = request.geometry.empty() ? 0 : request.geometry.front();
    parakeet_coreml_diarizer * context = parakeet_coreml_diarizer_init_with_units(
        request.artifact_path.c_str(), pcm::to_string(request.units));
    if (context == nullptr || parakeet_coreml_diarizer_capacity(context) != capacity) {
        PARAKEET_LOG_WARN("parakeet: failed to load Core ML diarizer at '%s'; using ggml\n",
                          request.artifact_path.c_str());
        parakeet_coreml_diarizer_free(context);
        result.error = pcm::Error::deterministic("failed to load Core ML diarizer", "parakeet", capacity);
        pcm::generation::end_load(request, result);
        return result;
    }
    PARAKEET_LOG_INFO("parakeet: Core ML diarizer loaded from '%s'\n", request.artifact_path.c_str());
    result.model = std::shared_ptr<void>(context, [](void * p) {
        parakeet_coreml_diarizer_free(static_cast<parakeet_coreml_diarizer *>(p));
    });
    pcm::generation::end_load(request, result);
    return result;
}

// Failed, retired or never routable (disabled, absent): the next covering
// capacity or ggml serves the pass.
bool unusable(pcm::SlotState state) { return pcm::is_terminal(state); }

}  // namespace

struct CoremlDiarizers::State {
    struct Capacity {
        int frames = 0;
        std::shared_ptr<pcm::Slot> slot;
    };
    std::string gguf_path;
    bool strict = false;               // use = require
    std::vector<Capacity> capacities;  // ascending
    std::mutex mutex;                  // label and call stats
    std::string label;
    CoremlDiarizerCallStats stats;
};

namespace {

std::vector<CoremlDiarizers::Capacity> prebuilt_only(const std::vector<int> & capacities) {
    std::vector<CoremlDiarizers::Capacity> out;
    for (int capacity : capacities) out.push_back(CoremlDiarizers::Capacity{capacity, true, nullptr});
    return out;
}

SidecarRequest diarizer_request(const std::string & gguf_path, const CoremlDiarizers::Capacity & c,
                                const pcm::SourceStat & source) {
    SidecarRequest request;
    request.stage = "diarizer-" + std::to_string(c.frames);
    request.path = coreml_diarizer_sidecar_path(gguf_path, c.frames);
    request.present = c.prebuilt;
    request.stage_default = pcm::ComputeUnits::cpu_and_ane;
    request.load_fn = &load_diarizer;
    request.geometry = {c.frames};
    request.generated = c.generated;
    request.source = source;
    return request;
}

}  // namespace

CoremlDiarizers::CoremlDiarizers(std::string gguf_path, std::vector<int> capacities,
                                 CoremlSidecars & sidecars)
    : CoremlDiarizers(std::move(gguf_path), prebuilt_only(capacities), sidecars) {}

CoremlDiarizers::CoremlDiarizers(std::string gguf_path, std::vector<Capacity> capacities,
                                 CoremlSidecars & sidecars)
    : state_(std::make_unique<State>()) {
    state_->gguf_path = std::move(gguf_path);
    state_->strict = sidecars.resolved().strict();
    std::sort(capacities.begin(), capacities.end(),
              [](const Capacity & a, const Capacity & b) { return a.frames < b.frames; });
    pcm::SourceStat source;
    for (const Capacity & c : capacities) {
        if (c.generated != nullptr && !c.prebuilt && source.path.empty()) {
            pcm::generation::source_stat(state_->gguf_path, source);
        }
    }
    for (const Capacity & c : capacities) {
        SidecarRequest request = diarizer_request(state_->gguf_path, c, source);
        // The smallest capacity is the minimal set; larger ones are
        // requested by the call that needs them (or by prewarm = all).
        request.minimal = state_->capacities.empty();
        state_->capacities.push_back(State::Capacity{c.frames, sidecars.add(request)});
    }
}

CoremlDiarizers::~CoremlDiarizers() = default;

bool CoremlDiarizers::covers(int frames) {
    if (frames <= 0) return false;
    for (const State::Capacity & c : state_->capacities) {
        // Under strict a failed capacity still covers, so run() refuses the pass.
        if (c.frames >= frames && (state_->strict || !unusable(c.slot->state()))) return true;
    }
    return false;
}

bool CoremlDiarizers::prepare(int frames) {
    if (frames <= 0) return false;
    for (const State::Capacity & c : state_->capacities) {
        if (c.frames < frames || unusable(c.slot->state())) continue;
        c.slot->request(pcm::Priority::current);
        return c.slot->state() == pcm::SlotState::ready;
    }
    return false;
}

int CoremlDiarizers::run(int valid_frames,
                         const parakeet_coreml_rows * inputs, int n_inputs,
                         const parakeet_coreml_rows_out * outputs, int n_outputs) {
    if (valid_frames <= 0) return kNoSidecar;
    const bool strict = state_->strict;
    // Strict: why the covering capacities tried so far could not serve the pass.
    std::optional<pcm::CoremlError> refused;
    for (const State::Capacity & c : state_->capacities) {
        if (c.frames < valid_frames) continue;
        // Strict acquisitions of a failed capacity throw; others skip it.
        if (!strict && unusable(c.slot->state())) continue;
        // One acquisition per pass (a chunk or block is the unit).
        pcm::Acquired acquired;
        try {
            acquired = c.slot->acquire(pcm::Demand::now);
        } catch (const pcm::CoremlError & error) {
            // A failed capacity gives way to the next covering one; a timeout
            // or a memory deferral refuses the pass at once.
            if (!pcm::is_terminal(error.state)) throw;
            if (!refused) refused = error;
            continue;
        }
        if (!acquired.model) {
            // Failed while this pass waited (wait = call): try the next capacity.
            if (unusable(acquired.state)) continue;
            std::lock_guard<std::mutex> lock(state_->mutex);
            ++state_->stats.not_ready;
            return kNoSidecar;
        }
        auto * context = acquired.model->as<parakeet_coreml_diarizer>();
        int status = 0;
        {
            auto prediction = acquired.model->lock_for_prediction();
            status = parakeet_coreml_diarizer_run(context, valid_frames, inputs, n_inputs,
                                                  outputs, n_outputs);
        }
        if (status != 0) {
            const std::string reason = "prediction failed (rc=" + std::to_string(status) + ")";
            PARAKEET_LOG_WARN("parakeet: Core ML diarizer (%d frames) failed (rc=%d); "
                              "quarantining it%s\n", c.frames, status, strict ? "" : " and using ggml");
            c.slot->retire(reason);
            if (strict) {
                throw pcm::CoremlError("coreml: " + c.slot->spec().stage + " " + reason +
                                           " and Core ML is required",
                                       pcm::SlotState::retired, pcm::ErrorClass::transient);
            }
            return status;
        }
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->label = parakeet_coreml_diarizer_label(context);
        ++state_->stats.coreml;
        state_->stats.label = state_->label;
        return 0;
    }
    if (refused) throw *refused;
    return kNoSidecar;
}

std::string CoremlDiarizers::label() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->label.empty() ? parakeet_coreml_diarizer_label(nullptr) : state_->label;
}

std::vector<int> CoremlDiarizers::capacities() const {
    std::vector<int> result;
    result.reserve(state_->capacities.size());
    for (const State::Capacity & c : state_->capacities) result.push_back(c.frames);
    return result;
}

void CoremlDiarizers::reset_call_stats() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->stats = CoremlDiarizerCallStats{};
}

CoremlDiarizerCallStats CoremlDiarizers::call_stats() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->stats;
}

const std::vector<CoremlDiarizerTemplate> & nemotron_diarizer_templates() {
    static const std::vector<CoremlDiarizerTemplate> templates = {
        {358, "nemotron-diar-f358"}, {745, "nemotron-diar-f745"}, {1125, "nemotron-diar-f1125"}};
    return templates;
}

std::unique_ptr<CoremlDiarizers> discover_coreml_diarizers(const std::string & gguf_path,
                                                           CoremlSidecars & sidecars,
                                                           const std::vector<CoremlDiarizerTemplate> & templates) {
    std::vector<CoremlDiarizers::Capacity> capacities = prebuilt_only(scan_capacities(gguf_path));
    // Templates are probed (the GGUF's metadata and tensor shapes) only when
    // this instance can generate at all. The candidates are read only while
    // the slots are registered below.
    std::vector<pcm::generation::Candidate> probed;
    if (!templates.empty() && sidecars.generation_status().supported) {
        std::vector<std::string> ids;
        for (const CoremlDiarizerTemplate & t : templates) ids.emplace_back(t.template_id);
        probed = pcm::generation::probe(ids, gguf_path, sidecars.resolved());
        pcm::SourceStat source;
        pcm::generation::source_stat(gguf_path, source);
        for (size_t i = 0; i < templates.size() && i < probed.size(); ++i) {
            auto found = std::find_if(capacities.begin(), capacities.end(), [&](const CoremlDiarizers::Capacity & c) {
                return c.frames == templates[i].frames;
            });
            if (found != capacities.end()) continue;  // an adjacent prebuilt bundle wins
            const CoremlDiarizers::Capacity c{templates[i].frames, false, &probed[i]};
            const pcm::Plan plan = sidecars.plan(diarizer_request(gguf_path, c, source));
            if (plan.route != pcm::Route::generate) {
                PARAKEET_LOG_INFO("parakeet: no Core ML diarizer for %d frames is generated (%s: %s)\n",
                                  c.frames, probed[i].template_id.c_str(),
                                  probed[i].matched ? plan.reason.c_str() : probed[i].reason.c_str());
                continue;
            }
            capacities.push_back(c);
        }
    }
    if (capacities.empty()) return nullptr;
    return std::make_unique<CoremlDiarizers>(gguf_path, std::move(capacities), sidecars);
}

}
