using System.Buffers;
using System.Diagnostics;

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

using PersonaEngine.Lib.Audio;
using PersonaEngine.Lib.Configuration;
using PersonaEngine.Lib.TTS.Synthesis;

namespace PersonaEngine.Lib.TTS.RVC;

public class RVCFilter : IAudioFilter, IDisposable
{
    private const int ProcessingSampleRate = 16000;

    private const int OutputSampleRate = 32000;

    private const int FinalSampleRate = 24000;

    private const int MaxInputDuration = 30; // seconds

    private readonly SemaphoreSlim _initLock = new(1, 1);

    private readonly ILogger<RVCFilter> _logger;

    private readonly IModelProvider _modelProvider;

    private readonly IDisposable? _optionsChangeRegistration;

    private readonly IOptionsMonitor<RVCFilterOptions> _optionsMonitor;

    private readonly IRVCVoiceProvider _rvcVoiceProvider;

    private RVCFilterOptions _currentOptions;

    private bool _disposed;

    private IF0Predictor? _f0Predictor;

    private OnnxRVC? _rvcModel;

    public RVCFilter(
        IOptionsMonitor<RVCFilterOptions> optionsMonitor,
        IModelProvider                    modelProvider,
        IRVCVoiceProvider                 rvcVoiceProvider,
        ILogger<RVCFilter>                logger)
    {
        _optionsMonitor   = optionsMonitor ?? throw new ArgumentNullException(nameof(optionsMonitor));
        _modelProvider    = modelProvider;
        _rvcVoiceProvider = rvcVoiceProvider;
        _logger           = logger ?? throw new ArgumentNullException(nameof(logger));
        _currentOptions   = optionsMonitor.CurrentValue;

        _ = InitializeAsync(_currentOptions);

        // Register for options changes
        _optionsChangeRegistration = _optionsMonitor.OnChange(OnOptionsChanged);
    }

    public void Process(AudioSegment audioSegment)
    {
        if ( _disposed )
        {
            throw new ObjectDisposedException(nameof(RVCFilter));
        }

        if ( _rvcModel == null || _f0Predictor == null ||
             audioSegment?.AudioData == null || audioSegment.AudioData.Length == 0 )
        {
            return;
        }

        // Get the latest options for processing
        var options            = _currentOptions;
        var originalSampleRate = audioSegment.SampleRate;

        if ( !options.Enabled )
        {
            return;
        }

        // Start timing
        var stopwatch = Stopwatch.StartNew();

        // Step 1: Resample input to processing sample rate
        var resampleRatioToProcessing = (int)Math.Ceiling((double)ProcessingSampleRate / originalSampleRate);
        var resampledInputSize        = audioSegment.AudioData.Length * resampleRatioToProcessing;

        var resampledInput = ArrayPool<float>.Shared.Rent(resampledInputSize);
        try
        {
            var inputSampleCount = AudioConverter.ResampleFloat(
                                                                audioSegment.AudioData,
                                                                resampledInput,
                                                                1,
                                                                (uint)originalSampleRate,
                                                                ProcessingSampleRate);

            // Step 2: Process with RVC model
            var maxInputSamples  = OutputSampleRate * MaxInputDuration;
            var outputBufferSize = maxInputSamples + 2 * options.HopSize;

            var processingBuffer = ArrayPool<float>.Shared.Rent(outputBufferSize);
            try
            {
                var processedSampleCount = _rvcModel.ProcessAudio(
                                                                  resampledInput.AsMemory(0, inputSampleCount),
                                                                  processingBuffer,
                                                                  _f0Predictor,
                                                                  options.SpeakerId,
                                                                  options.F0UpKey);

                // Step 3: Resample to original sample rate
                var resampleRatioToOutput = (int)Math.Ceiling((double)originalSampleRate / OutputSampleRate);
                var finalOutputSize       = processedSampleCount * resampleRatioToOutput;

                var resampledOutput = ArrayPool<float>.Shared.Rent(finalOutputSize);
                try
                {
                    var finalSampleCount = AudioConverter.ResampleFloat(
                                                                        processingBuffer.AsMemory(0, processedSampleCount),
                                                                        resampledOutput,
                                                                        1,
                                                                        OutputSampleRate,
                                                                        (uint)originalSampleRate);

                    // Need one allocation for the final output buffer since AudioSegment keeps this reference
                    var finalBuffer = new float[finalSampleCount];
                    Array.Copy(resampledOutput, finalBuffer, finalSampleCount);

                    audioSegment.AudioData  = finalBuffer.AsMemory();
                    audioSegment.SampleRate = FinalSampleRate;
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(resampledOutput);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(processingBuffer);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(resampledInput);
        }

        // Stop timing after processing is complete
        stopwatch.Stop();
        var processingTime = stopwatch.Elapsed.TotalSeconds;

        // Calculate final audio duration (based on the processed audio)
        var finalAudioDuration = audioSegment.AudioData.Length / (double)FinalSampleRate;

        // Calculate real-time factor
        var realTimeFactor = finalAudioDuration / processingTime;

        // Log the results using ILogger
        _logger.LogInformation("Generated {AudioDuration:F2}s audio in {ProcessingTime:F2}s (x{RealTimeFactor:F2} real-time)",
                               finalAudioDuration, processingTime, realTimeFactor);
    }

    public int Priority => 100;

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if ( _disposed )
        {
            return;
        }

        if ( disposing )
        {
            _optionsChangeRegistration?.Dispose();
            DisposeResources();
        }

        _disposed = true;
    }

    private async void OnOptionsChanged(RVCFilterOptions newOptions)
    {
        if ( _disposed )
        {
            return;
        }

        if ( ShouldReinitialize(newOptions) )
        {
            DisposeResources();
            await InitializeAsync(newOptions);
        }

        _currentOptions = newOptions;
    }

    private bool ShouldReinitialize(RVCFilterOptions newOptions)
    {
        return _currentOptions.DefaultVoice != newOptions.DefaultVoice ||
               _currentOptions.HopSize != newOptions.HopSize;
    }

    private async ValueTask InitializeAsync(RVCFilterOptions options)
    {
        await _initLock.WaitAsync();
        try
        {
            var crepeModel  = await _modelProvider.GetModelAsync(Synthesis.ModelType.RVCCrepeTiny);
            var hubertModel = await _modelProvider.GetModelAsync(Synthesis.ModelType.RVCHubert);
            var rvcModel    = await _rvcVoiceProvider.GetVoiceAsync(options.DefaultVoice);

            // _f0Predictor = new CrepeOnnx(crepeModel.Path);
            // _f0Predictor = new CrepeOnnxSimd(crepeModel.Path);
            _f0Predictor = new ACFMethod(512, 16000);

            _rvcModel = new OnnxRVC(
                                    rvcModel,
                                    options.HopSize,
                                    hubertModel.Path);
        }
        finally
        {
            _initLock.Release();
        }
    }

    private void DisposeResources()
    {
        _rvcModel?.Dispose();
        _rvcModel = null;

        _f0Predictor?.Dispose();
        _f0Predictor = null;
    }
}