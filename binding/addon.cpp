// addon.cpp — the Node binding.
//
// Two things matter for this to be worth having:
//
//  1. No copying on the hot path. Vectors arrive as a Float32Array and are
//     read straight out of its backing buffer; nothing is converted element
//     by element. A conversion loop in JavaScript would cost more than the
//     search it is feeding.
//
//  2. Long operations release the event loop. Building an index over a
//     million vectors takes minutes, and doing that on the main thread would
//     freeze the entire process. addBatch runs on a worker thread and
//     resolves a promise.
#include <napi.h>

#include <memory>
#include <string>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/hnsw.hpp"

namespace {

using namespace vecsearch;

Metric parseMetric(const std::string& name) {
    if (name == "l2") return Metric::L2;
    if (name == "ip" || name == "cosine" || name == "inner_product") return Metric::InnerProduct;
    throw std::invalid_argument("metric must be \"l2\" or \"ip\" (got \"" + name + "\")");
}

/// Reads a Float32Array without copying. Any other type is rejected loudly:
/// silently accepting a plain Array and converting it would hide a
/// performance problem rather than solve it.
const float* float32Data(const Napi::Value& value, std::size_t expected, const char* what) {
    if (!value.IsTypedArray() || value.As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
        throw Napi::TypeError::New(value.Env(), std::string(what) + " must be a Float32Array");
    }
    auto array = value.As<Napi::Float32Array>();
    if (expected && array.ElementLength() != expected) {
        throw Napi::TypeError::New(
            value.Env(), std::string(what) + " has " + std::to_string(array.ElementLength()) +
                             " elements but the index expects " + std::to_string(expected));
    }
    return array.Data();
}

}  // namespace

class VecIndex : public Napi::ObjectWrap<VecIndex> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    VecIndex(const Napi::CallbackInfo& info);

private:
    Napi::Value Add(const Napi::CallbackInfo& info);
    Napi::Value AddBatch(const Napi::CallbackInfo& info);
    Napi::Value Search(const Napi::CallbackInfo& info);
    Napi::Value BruteForce(const Napi::CallbackInfo& info);
    Napi::Value MarkDeleted(const Napi::CallbackInfo& info);
    Napi::Value Save(const Napi::CallbackInfo& info);
    Napi::Value Stats(const Napi::CallbackInfo& info);
    Napi::Value Size(const Napi::CallbackInfo& info);
    Napi::Value Close(const Napi::CallbackInfo& info);

    /// Throws a clear JS error instead of dereferencing a released index.
    HnswIndex& live(Napi::Env env) {
        if (!index_) throw Napi::Error::New(env, "This index has been closed.");
        return *index_;
    }

    static Napi::Value Load(const Napi::CallbackInfo& info);
    static Napi::Value OpenMapped(const Napi::CallbackInfo& info);
    static Napi::Object WrapIndex(Napi::Env env, std::unique_ptr<HnswIndex> index);

    std::unique_ptr<HnswIndex> index_;

    /// Builds currently running on worker threads. close() refuses while this
    /// is non-zero: the worker holds a raw pointer into the index.
    int pending_builds_ = 0;

    friend class BuildWorker;
};

/// Runs a batch insert off the main thread.
class BuildWorker : public Napi::AsyncWorker {
public:
    BuildWorker(Napi::Env env, VecIndex* owner, Napi::Object ownerObject, HnswIndex* index,
                Napi::Float32Array vectors, std::vector<label_t> labels, std::size_t count, int threads)
        : Napi::AsyncWorker(env),
          deferred_(Napi::Promise::Deferred::New(env)),
          owner_(owner),
          // Keeps the JS object alive until the build finishes. Without it,
          // dropping the last reference mid-build would let the garbage
          // collector free the index under the worker thread.
          owner_ref_(Napi::Persistent(ownerObject)),
          index_(index),
          labels_(std::move(labels)),
          count_(count),
          threads_(threads),
          // The reference keeps the ArrayBuffer alive while the worker runs;
          // without it V8 could collect the data mid-build.
          ref_(Napi::Persistent(vectors)),
          data_(vectors.Data()) {}

    void Execute() override {
        try {
            index_->addBatch(data_, labels_.data(), count_, threads_);
        } catch (const std::exception& e) {
            SetError(e.what());
        }
    }

    // Both run back on the main thread, so touching owner_ needs no lock.
    void OnOK() override {
        --owner_->pending_builds_;
        deferred_.Resolve(Napi::Number::New(Env(), static_cast<double>(index_->size())));
    }
    void OnError(const Napi::Error& e) override {
        --owner_->pending_builds_;
        deferred_.Reject(e.Value());
    }

    Napi::Promise Promise() { return deferred_.Promise(); }

private:
    Napi::Promise::Deferred deferred_;
    VecIndex* owner_;
    Napi::ObjectReference owner_ref_;
    HnswIndex* index_;
    std::vector<label_t> labels_;
    std::size_t count_;
    int threads_;
    Napi::Reference<Napi::Float32Array> ref_;
    const float* data_;
};

Napi::Object VecIndex::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "VecIndex", {
        InstanceMethod("add", &VecIndex::Add),
        InstanceMethod("addBatch", &VecIndex::AddBatch),
        InstanceMethod("search", &VecIndex::Search),
        InstanceMethod("bruteForce", &VecIndex::BruteForce),
        InstanceMethod("markDeleted", &VecIndex::MarkDeleted),
        InstanceMethod("save", &VecIndex::Save),
        InstanceMethod("stats", &VecIndex::Stats),
        InstanceMethod("close", &VecIndex::Close),
        InstanceAccessor("size", &VecIndex::Size, nullptr),
        StaticMethod("load", &VecIndex::Load),
        StaticMethod("openMapped", &VecIndex::OpenMapped),
    });

    auto* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("VecIndex", func);
    return exports;
}

VecIndex::VecIndex(const Napi::CallbackInfo& info) : Napi::ObjectWrap<VecIndex>(info) {
    Napi::Env env = info.Env();

    // Constructed internally by load()/openMapped(), which attach the index
    // afterwards via the external pointer.
    if (info.Length() == 1 && info[0].IsExternal()) {
        index_.reset(info[0].As<Napi::External<HnswIndex>>().Data());
        return;
    }

    if (info.Length() < 1 || !info[0].IsObject()) {
        throw Napi::TypeError::New(env, "new VecIndex({ dim, metric, M, efConstruction, capacity })");
    }
    Napi::Object opts = info[0].As<Napi::Object>();

    // Everything from here is wrapped: a std::exception escaping into V8 is
    // not an error the JavaScript side can catch, it terminates the process.
    try {
        IndexConfig cfg;
        if (!opts.Has("dim")) throw std::invalid_argument("dim is required");
        cfg.dim = opts.Get("dim").As<Napi::Number>().Uint32Value();
        cfg.metric = parseMetric(opts.Has("metric") ? opts.Get("metric").As<Napi::String>().Utf8Value() : "l2");
        if (opts.Has("M")) cfg.M = opts.Get("M").As<Napi::Number>().Uint32Value();
        if (opts.Has("efConstruction")) cfg.ef_construction = opts.Get("efConstruction").As<Napi::Number>().Uint32Value();
        if (opts.Has("capacity")) cfg.capacity = opts.Get("capacity").As<Napi::Number>().Int64Value();
        if (opts.Has("seed")) cfg.seed = opts.Get("seed").As<Napi::Number>().Int64Value();

        index_ = std::make_unique<HnswIndex>(cfg);
    } catch (const Napi::Error&) {
        throw;
    } catch (const std::exception& e) {
        throw Napi::Error::New(env, e.what());
    }
}

Napi::Value VecIndex::Add(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) throw Napi::TypeError::New(env, "add(vector, label)");

    const float* data = float32Data(info[0], live(env).dim(), "vector");
    const auto label = static_cast<label_t>(info[1].As<Napi::Number>().Int64Value());

    try {
        return Napi::Number::New(env, static_cast<double>(live(env).add(data, label)));
    } catch (const std::exception& e) {
        throw Napi::Error::New(env, e.what());
    }
}

Napi::Value VecIndex::AddBatch(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) throw Napi::TypeError::New(env, "addBatch(vectors, labels?, options?)");

    auto vectors = info[0].As<Napi::Float32Array>();
    if (!info[0].IsTypedArray() || info[0].As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
        throw Napi::TypeError::New(env, "vectors must be a Float32Array holding count * dim values");
    }

    const std::size_t dim = live(env).dim();
    if (vectors.ElementLength() % dim != 0) {
        throw Napi::TypeError::New(env, "the Float32Array length is not a multiple of dim");
    }
    const std::size_t count = vectors.ElementLength() / dim;

    std::vector<label_t> labels(count);
    if (info.Length() > 1 && info[1].IsArray()) {
        auto arr = info[1].As<Napi::Array>();
        if (arr.Length() != count) throw Napi::TypeError::New(env, "labels.length must equal the number of vectors");
        for (std::size_t i = 0; i < count; ++i) {
            labels[i] = static_cast<label_t>(arr.Get(i).As<Napi::Number>().Int64Value());
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) labels[i] = static_cast<label_t>(live(env).size() + i);
    }

    int threads = 1;
    if (info.Length() > 2 && info[2].IsObject()) {
        Napi::Object opts = info[2].As<Napi::Object>();
        if (opts.Has("threads")) threads = opts.Get("threads").As<Napi::Number>().Int32Value();
    }

    if (threads > 1 && live(env).size() + count > 0) {
        // A threaded build may not grow the index, so make sure the space is
        // there before the workers start.
        try {
            live(env).reserve(live(env).size() + count);
        } catch (const std::exception& e) {
            throw Napi::Error::New(env, e.what());
        }
    }

    auto* worker = new BuildWorker(env, this, info.This().As<Napi::Object>(), &live(env),
                                   vectors, std::move(labels), count, threads);
    ++pending_builds_;
    auto promise = worker->Promise();
    worker->Queue();
    return promise;
}

Napi::Value VecIndex::Search(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2) throw Napi::TypeError::New(env, "search(query, k, ef?)");

    const float* query = float32Data(info[0], live(env).dim(), "query");
    const auto k = static_cast<std::size_t>(info[1].As<Napi::Number>().Uint32Value());
    const std::size_t ef = info.Length() > 2 && info[2].IsNumber()
                               ? info[2].As<Napi::Number>().Uint32Value()
                               : std::max<std::size_t>(k * 4, 32);

    std::vector<SearchResult> results;
    try {
        results = live(env).search(query, k, ef);
    } catch (const std::exception& e) {
        throw Napi::Error::New(env, e.what());
    }

    Napi::Array out = Napi::Array::New(env, results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        Napi::Object item = Napi::Object::New(env);
        item.Set("label", Napi::Number::New(env, static_cast<double>(results[i].label)));
        item.Set("distance", Napi::Number::New(env, results[i].distance));
        out.Set(i, item);
    }
    return out;
}

Napi::Value VecIndex::BruteForce(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const float* query = float32Data(info[0], live(env).dim(), "query");
    const auto k = static_cast<std::size_t>(info[1].As<Napi::Number>().Uint32Value());

    const auto results = live(info.Env()).bruteForce(query, k);
    Napi::Array out = Napi::Array::New(env, results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        Napi::Object item = Napi::Object::New(env);
        item.Set("label", Napi::Number::New(env, static_cast<double>(results[i].label)));
        item.Set("distance", Napi::Number::New(env, results[i].distance));
        out.Set(i, item);
    }
    return out;
}

Napi::Value VecIndex::MarkDeleted(const Napi::CallbackInfo& info) {
    const auto label = static_cast<label_t>(info[0].As<Napi::Number>().Int64Value());
    try {
        return Napi::Boolean::New(info.Env(), live(info.Env()).markDeleted(label));
    } catch (const std::exception& e) {
        throw Napi::Error::New(info.Env(), e.what());
    }
}

Napi::Value VecIndex::Save(const Napi::CallbackInfo& info) {
    const std::string path = info[0].As<Napi::String>().Utf8Value();
    try {
        live(info.Env()).save(path);
    } catch (const std::exception& e) {
        throw Napi::Error::New(info.Env(), e.what());
    }
    return info.Env().Undefined();
}

Napi::Value VecIndex::Size(const Napi::CallbackInfo& info) {
    // A closed index reports size 0 rather than throwing from a property read.
    return Napi::Number::New(info.Env(), index_ ? static_cast<double>(index_->size()) : 0.0);
}

/**
 * Releases the index — and, for a memory-mapped one, the file mapping.
 *
 * JavaScript frees native memory whenever the garbage collector gets round
 * to it, which may be never in a short script. That is merely wasteful on
 * Linux, but on Windows a mapped file cannot be deleted or overwritten
 * while the mapping exists, so "delete the index file" fails with EBUSY
 * until the GC happens to run. close() makes the release deterministic.
 */
Napi::Value VecIndex::Close(const Napi::CallbackInfo& info) {
    if (pending_builds_ > 0) {
        throw Napi::Error::New(info.Env(), "Cannot close the index while addBatch() is still running. Await it first.");
    }
    index_.reset();
    return info.Env().Undefined();
}

Napi::Value VecIndex::Stats(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const IndexStats s = live(env).stats();
    Napi::Object out = Napi::Object::New(env);
    out.Set("count", Napi::Number::New(env, static_cast<double>(s.count)));
    out.Set("deleted", Napi::Number::New(env, static_cast<double>(s.deleted)));
    out.Set("dim", Napi::Number::New(env, static_cast<double>(s.dim)));
    out.Set("paddedDim", Napi::Number::New(env, static_cast<double>(s.padded_dim)));
    out.Set("M", Napi::Number::New(env, static_cast<double>(s.M)));
    out.Set("efConstruction", Napi::Number::New(env, static_cast<double>(s.ef_construction)));
    out.Set("levels", Napi::Number::New(env, s.max_level + 1));
    out.Set("memoryBytes", Napi::Number::New(env, static_cast<double>(s.memory_bytes)));
    out.Set("distanceCalls", Napi::Number::New(env, static_cast<double>(s.distance_calls)));
    out.Set("kernel", Napi::String::New(env, s.kernel));
    out.Set("readOnly", Napi::Boolean::New(env, live(env).readOnly()));
    return out;
}

Napi::Object VecIndex::WrapIndex(Napi::Env env, std::unique_ptr<HnswIndex> index) {
    auto* constructor = env.GetInstanceData<Napi::FunctionReference>();
    Napi::External<HnswIndex> external = Napi::External<HnswIndex>::New(env, index.get());
    Napi::Object obj = constructor->New({external});
    index.release();  // ownership passed to the wrapper
    return obj;
}

Napi::Value VecIndex::Load(const Napi::CallbackInfo& info) {
    const std::string path = info[0].As<Napi::String>().Utf8Value();
    try {
        return WrapIndex(info.Env(), std::make_unique<HnswIndex>(HnswIndex::load(path)));
    } catch (const std::exception& e) {
        throw Napi::Error::New(info.Env(), e.what());
    }
}

Napi::Value VecIndex::OpenMapped(const Napi::CallbackInfo& info) {
    const std::string path = info[0].As<Napi::String>().Utf8Value();
    try {
        return WrapIndex(info.Env(), std::make_unique<HnswIndex>(HnswIndex::openMapped(path)));
    } catch (const std::exception& e) {
        throw Napi::Error::New(info.Env(), e.what());
    }
}

static Napi::Value CpuInfo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Object out = Napi::Object::New(env);
    out.Set("avx2", Napi::Boolean::New(env, cpuFeatures().avx2));
    out.Set("fma", Napi::Boolean::New(env, cpuFeatures().fma));
    out.Set("kernel", Napi::String::New(env, activeKernelName(Metric::L2, 384)));
    return out;
}

static Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    exports.Set("cpuInfo", Napi::Function::New(env, CpuInfo));
    return VecIndex::Init(env, exports);
}

NODE_API_MODULE(vecsearch, InitAll)
