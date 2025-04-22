using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using PersonaEngine.Lib.ASR.VAD;

namespace PersonaEngine.Lib.ASR.Biometrics
{
    /// <summary>
    /// Concrete implementation of IVoiceBiometricService.
    /// </summary>
    public class VoiceBiometricService : IVoiceBiometricService
    {
        private readonly VoiceFeatureExtractor _featureExtractor;
        private readonly VoiceBiometricsEnrollmentService _enrollmentService;
        private readonly VoiceBiometricsMatcher _matcher;

        public VoiceBiometricService(
            VoiceFeatureExtractor featureExtractor,
            VoiceBiometricsEnrollmentService enrollmentService,
            VoiceBiometricsMatcher matcher)
        {
            _featureExtractor = featureExtractor;
            _enrollmentService = enrollmentService;
            _matcher = matcher;
        }

        public Task EnrollAsync(string userId, IEnumerable<IEnumerable<VadSegment>> samples, CancellationToken cancellationToken = default)
        {
            _enrollmentService.EnrollUser(userId, samples);
            return Task.CompletedTask;
        }

        public Task<bool> VerifyAsync(string userId, IEnumerable<VadSegment> sampleSegments, CancellationToken cancellationToken = default)
        {
            var result = _matcher.Verify(userId, sampleSegments);
            return Task.FromResult(result);
        }

        public Task<(string userId, float similarity)?> IdentifyAsync(IEnumerable<VadSegment> sampleSegments, CancellationToken cancellationToken = default)
        {
            var result = _matcher.Identify(sampleSegments);
            return Task.FromResult(result);
        }
    }
}