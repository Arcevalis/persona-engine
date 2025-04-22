using PersonaEngine.Lib.Audio;
using PersonaEngine.Lib.ASR.VAD;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace PersonaEngine.Lib.ASR.Biometrics
{
    public class VoiceBiometricsAudioPreprocessor
    {
        private readonly IVadDetector _vadDetector;

        public VoiceBiometricsAudioPreprocessor(IVadDetector vadDetector)
        {
            _vadDetector = vadDetector;
        }

        /// <summary>
        /// Captures audio from the source and returns speech segments for biometrics processing.
        /// </summary>
        public async Task<List<VadSegment>> CaptureAndPreprocessAsync(
            IAudioSource audioSource,
            CancellationToken cancellationToken = default)
        {
            // Validate audio source (mono, 16kHz)
            ValidateSource(audioSource);

            // Detect speech segments using VAD asynchronously
            var segments = new List<VadSegment>();
            await foreach (var segment in _vadDetector.DetectSegmentsAsync(audioSource, cancellationToken))
            {
                segments.Add(segment);
            }

            // Return only the speech segments
            return segments;
        }

        private static void ValidateSource(IAudioSource source)
        {
            if (source.ChannelCount != 1)
            {
                throw new NotSupportedException("Only mono-channel audio is supported. Consider one channel aggregation on the audio source.");
            }

            if (source.SampleRate != 16000)
            {
                throw new NotSupportedException("Only 16 kHz audio is supported. Consider resampling before calling this preprocessor.");
            }
        }
    }
}