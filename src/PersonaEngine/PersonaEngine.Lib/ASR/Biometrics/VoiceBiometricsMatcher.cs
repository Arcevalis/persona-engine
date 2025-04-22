using System;
using System.Collections.Generic;
using System.Linq;

namespace PersonaEngine.Lib.ASR.Biometrics
{
    /// <summary>
    /// Provides verification and identification for voice biometrics.
    /// </summary>
    public class VoiceBiometricsMatcher
    {
        private readonly VoiceFeatureExtractor _featureExtractor;
        private readonly VoiceBiometricsEnrollmentService _enrollmentService;
        private readonly float _similarityThreshold;

        public VoiceBiometricsMatcher(
            VoiceFeatureExtractor featureExtractor,
            VoiceBiometricsEnrollmentService enrollmentService,
            float similarityThreshold = 0.8f)
        {
            _featureExtractor = featureExtractor;
            _enrollmentService = enrollmentService;
            _similarityThreshold = similarityThreshold;
        }

        /// <summary>
        /// Verifies if the provided sample matches the claimed user's voiceprint.
        /// </summary>
        public bool Verify(string userId, IEnumerable<PersonaEngine.Lib.ASR.VAD.VadSegment> sampleSegments)
        {
            var enrolledVoiceprint = _enrollmentService.GetVoiceprint(userId);
            if (enrolledVoiceprint == null)
                return false;

            var features = _featureExtractor.ExtractFeatures(sampleSegments);
            if (features.Count == 0)
                return false;

            // Aggregate features from the sample (mean vector)
            var sampleVoiceprint = MeanVector(features);

            var similarity = CosineSimilarity(enrolledVoiceprint, sampleVoiceprint);
            return similarity >= _similarityThreshold;
        }

        /// <summary>
        /// Identifies the user whose voiceprint best matches the provided sample.
        /// Returns the userId and similarity score, or null if no match exceeds the threshold.
        /// </summary>
        public (string userId, float similarity)? Identify(IEnumerable<PersonaEngine.Lib.ASR.VAD.VadSegment> sampleSegments)
        {
            var features = _featureExtractor.ExtractFeatures(sampleSegments);
            if (features.Count == 0)
                return null;

            var sampleVoiceprint = MeanVector(features);

            string? bestUser = null;
            float bestSimilarity = float.MinValue;

            foreach (var kvp in _enrollmentService.GetAllVoiceprints())
            {
                var similarity = CosineSimilarity(kvp.Value, sampleVoiceprint);
                if (similarity > bestSimilarity)
                {
                    bestSimilarity = similarity;
                    bestUser = kvp.Key;
                }
            }

            if (bestUser != null && bestSimilarity >= _similarityThreshold)
                return (bestUser, bestSimilarity);

            return null;
        }

        // Helper: Compute mean vector from a list of vectors
        private static float[] MeanVector(List<float[]> vectors)
        {
            var length = vectors[0].Length;
            var mean = new float[length];
            foreach (var vec in vectors)
                for (int i = 0; i < length; i++)
                    mean[i] += vec[i];
            for (int i = 0; i < length; i++)
                mean[i] /= vectors.Count;
            return mean;
        }

        // Helper: Cosine similarity between two vectors
        private static float CosineSimilarity(float[] v1, float[] v2)
        {
            float dot = 0, norm1 = 0, norm2 = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                dot += v1[i] * v2[i];
                norm1 += v1[i] * v1[i];
                norm2 += v2[i] * v2[i];
            }
            return (float)(dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2) + 1e-10));
        }
    }
}