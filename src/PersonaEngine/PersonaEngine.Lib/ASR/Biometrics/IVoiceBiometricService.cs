using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using PersonaEngine.Lib.ASR.VAD;

namespace PersonaEngine.Lib.ASR.Biometrics
{
    /// <summary>
    /// Interface for a unified voice biometric service.
    /// </summary>
    public interface IVoiceBiometricService
    {
        Task EnrollAsync(string userId, IEnumerable<IEnumerable<VadSegment>> samples, CancellationToken cancellationToken = default);
        Task<bool> VerifyAsync(string userId, IEnumerable<VadSegment> sampleSegments, CancellationToken cancellationToken = default);
        Task<(string userId, float similarity)?> IdentifyAsync(IEnumerable<VadSegment> sampleSegments, CancellationToken cancellationToken = default);
    }
}