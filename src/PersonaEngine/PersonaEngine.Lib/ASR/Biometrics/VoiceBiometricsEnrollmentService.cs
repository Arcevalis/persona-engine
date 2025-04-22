using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace PersonaEngine.Lib.ASR.Biometrics
{
    public class VoiceBiometricsEnrollmentService
    {
        private readonly string _storagePath;
        private readonly VoiceFeatureExtractor _featureExtractor;
        private Dictionary<string, float[]> _voiceprints;

        public VoiceBiometricsEnrollmentService(VoiceFeatureExtractor featureExtractor, string storagePath = "voiceprints.json")
        {
            _featureExtractor = featureExtractor;
            _storagePath = storagePath;
            _voiceprints = LoadVoiceprints();
        }

        public void EnrollUser(string userId, IEnumerable<IEnumerable<PersonaEngine.Lib.ASR.VAD.VadSegment>> samples)
        {
            var allFeatureVectors = new List<float[]>();

            foreach (var sampleSegments in samples)
            {
                var features = _featureExtractor.ExtractFeatures(sampleSegments);
                allFeatureVectors.AddRange(features);
            }

            if (allFeatureVectors.Count == 0)
                throw new InvalidOperationException("No features extracted from provided samples.");

            var featureLength = allFeatureVectors[0].Length;
            var meanVoiceprint = new float[featureLength];

            foreach (var vec in allFeatureVectors)
            {
                for (int i = 0; i < featureLength; i++)
                    meanVoiceprint[i] += vec[i];
            }
            for (int i = 0; i < featureLength; i++)
                meanVoiceprint[i] /= allFeatureVectors.Count;

            _voiceprints[userId] = meanVoiceprint;
            SaveVoiceprints();
        }

        public float[]? GetVoiceprint(string userId)
        {
            _voiceprints.TryGetValue(userId, out var voiceprint);
            return voiceprint;
        }

        public IReadOnlyDictionary<string, float[]> GetAllVoiceprints()
        {
            return _voiceprints;
        }

        private Dictionary<string, float[]> LoadVoiceprints()
        {
            if (!File.Exists(_storagePath))
                return new Dictionary<string, float[]>();

            var json = File.ReadAllText(_storagePath);
            return JsonSerializer.Deserialize<Dictionary<string, float[]>>(json) ?? new Dictionary<string, float[]>();
        }

        private void SaveVoiceprints()
        {
            var json = JsonSerializer.Serialize(_voiceprints);
            File.WriteAllText(_storagePath, json);
        }
    }
}