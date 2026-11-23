#include "moss/transcribe_coreml_slot.h"

#include "fit_util.h"
#include "parakeet_coreml_path.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <system_error>

namespace parakeet::moss::detail {
namespace {

constexpr size_t GEOMETRY_FIELDS = 5;  // loader kind, then the four window fields

bool is_directory(const std::string & path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

uint64_t ceil_scaled(uint64_t bytes, double factor) {
    return fitutil::sat_u64_from_double(std::ceil(static_cast<double>(bytes) * factor));
}

// A generated stage's charge from its template, the margin applied once to
// the whole envelope: the resident part times the margin, and the rest of
// generation::envelope_bytes as the transient.
CoremlCharge generated_charge(const pcm::generation::TemplateMemory & memory, pcm::ComputeUnits units) {
    CoremlCharge c;
    const uint64_t envelope = pcm::generation::envelope_bytes(memory, units);
    c.resident = std::min(envelope, ceil_scaled(pcm::generation::resident_bytes(memory), memory.margin));
    c.transient = envelope - c.resident;
    return c;
}

}  // namespace

std::vector<int64_t> transcribe_encoder_geometry(const TranscribeEncoderWindow & window) {
    return sidecar_geometry(SidecarLoader::transcribe_encoder,
                            {window.n_mels, window.frames, window.tokens, window.width});
}

bool transcribe_encoder_window_of(const std::vector<int64_t> & geometry, TranscribeEncoderWindow & window) {
    if (geometry.size() != GEOMETRY_FIELDS ||
        geometry[0] != static_cast<int64_t>(SidecarLoader::transcribe_encoder)) {
        return false;
    }
    window.n_mels = static_cast<int>(geometry[1]);
    window.frames = static_cast<int>(geometry[2]);
    window.tokens = static_cast<int>(geometry[3]);
    window.width = static_cast<int>(geometry[4]);
    return window.n_mels > 0 && window.frames > 0 && window.tokens > 0 && window.width > 0;
}

uint64_t transcribe_encoder_io_bytes(const TranscribeEncoderWindow & window) {
    using fitutil::sat_add;
    using fitutil::sat_mul;
    const uint64_t input = sat_mul((uint64_t) window.n_mels, (uint64_t) window.frames);
    const uint64_t output = sat_mul((uint64_t) window.tokens, (uint64_t) window.width);
    return sat_mul(sat_add(input, output), (uint64_t) sizeof(float));
}

pcm::generation::Candidate probe_transcribe_encoder_template(const CoremlSidecars & sidecars,
                                                             const std::string & gguf_path) {
    if (!pcm::generation::built() || !sidecars.resolved().enabled()) {
        pcm::generation::Candidate unprobed;
        unprobed.template_id = TRANSCRIBE_ENCODER_TEMPLATE;
        return unprobed;
    }
    std::vector<pcm::generation::Candidate> probed =
        pcm::generation::probe({TRANSCRIBE_ENCODER_TEMPLATE}, gguf_path, sidecars.resolved());
    return probed.empty() ? pcm::generation::Candidate{} : probed.front();
}

SidecarRequest transcribe_encoder_request(const std::string & gguf_path, const TranscribeEncoderWindow & window,
                                          const pcm::generation::Candidate * generated, pcm::LoadFn load_fn,
                                          const pcm::SourceStat & source) {
    SidecarRequest request;
    request.stage = TRANSCRIBE_ENCODER_STAGE;
    request.path = coreml_encoder_sidecar_path(gguf_path);
    request.present = is_directory(request.path);
    request.stage_default = pcm::ComputeUnits::all;
    if (!request.present && generated != nullptr && generated->matched && generated->default_units) {
        request.stage_default = *generated->default_units;
    }
    request.load_fn = load_fn;
    request.geometry = transcribe_encoder_geometry(window);
    // The encoder is the only sidecar, so the minimal set is the encoder.
    request.minimal = true;
    request.generated = generated;
    request.source = source;
    return request;
}

TranscribeEncoderFit transcribe_encoder_fit(const CoremlSidecars & sidecars, const std::string & gguf_path,
                                            const TranscribeEncoderWindow & window, uint64_t encoder_parameters) {
    TranscribeEncoderFit fit;
    const pcm::Resolved & resolved = sidecars.resolved();
    fit.enabled = resolved.enabled();
    if (!fit.enabled) {
        fit.disabled_reason = resolved.disabled_reason.empty() ? std::string("use=off") : resolved.disabled_reason;
        return fit;
    }
    const pcm::generation::Candidate candidate = probe_transcribe_encoder_template(sidecars, gguf_path);
    const SidecarRequest request = transcribe_encoder_request(gguf_path, window, &candidate, nullptr, {});
    const pcm::Plan plan = sidecars.plan(request);
    fit.route = plan.route;
    const pcm::ComputeUnits units = resolved.units_or(request.stage_default);
    if (plan.route == pcm::Route::prebuilt) {
        CoremlFitProgram program;
        program.weight_file_bytes = coreml_encoder_fit_route(request.path, true).weight_file_bytes;
        program.runtime_floor_bytes = fitutil::sat_mul(encoder_parameters, (uint64_t) 2);
        program.io_bytes = transcribe_encoder_io_bytes(window);
        program.units = fit_units(units);
        fit.charge = coreml_programs_charge({program});
    } else if (plan.route == pcm::Route::generate) {
        fit.charge = generated_charge(candidate.memory, units);
        fit.validated = true;
    }
    return fit;
}

}  // namespace parakeet::moss::detail
