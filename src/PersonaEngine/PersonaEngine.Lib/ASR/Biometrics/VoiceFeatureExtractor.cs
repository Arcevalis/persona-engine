using System.Collections.Generic;
using PersonaEngine.Lib.ASR.VAD;
using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Options;
using NWaves.Signals;

namespace PersonaEngine.Lib.ASR.Biometrics
{
    /// <summary>
    /// Extracts MFCC features from speech segments using NWaves.
    /// </summary>
    public class VoiceFeatureExtractor
    {
        private readonly MfccExtractor _mfccExtractor;

        public VoiceFeatureExtractor(int sampleRate = 16000, int featureCount = 13)
        {
            // Use MfccOptions to configure the extractor
            var options = new MfccOptions
            {
                SamplingRate = sampleRate,
                FeatureCount = featureCount,
                FrameDuration = 0.025, // 25 ms
                HopDuration = 0.01     // 10 ms
            };
            _mfccExtractor = new MfccExtractor(options);
        }

        /// <summary>
        /// Extracts MFCC features from each VadSegment.
        /// Returns a list of MFCC vectors (one per segment).
        /// </summary>
        public List<float[]> ExtractFeatures(IEnumerable<VadSegment> segments)
        {
            var features = new List<float[]>();
            foreach (var segment in segments)
            {
                // Convert float[] samples to NWaves DiscreteSignal
                var signal = new DiscreteSignal(segment.SampleRate, segment.Samples);

                // Extract MFCCs for the segment (returns a sequence of MFCC vectors per frame)
                var mfccVectors = _mfccExtractor.ComputeFrom(signal);

                // Aggregate MFCCs for the segment (e.g., mean vector)
                if (mfccVectors.Count > 0)
                {
                    var meanMfcc = new float[mfccVectors[0].Length];
                    foreach (var vec in mfccVectors)
                    {
                        for (int i = 0; i < vec.Length; i++)
                            meanMfcc[i] += vec[i];
                    }
                    for (int i = 0; i < meanMfcc.Length; i++)
                        meanMfcc[i] /= mfccVectors.Count;

                    features.Add(meanMfcc);
                }
            }
            return features;
        }
    }
}