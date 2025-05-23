using System.Globalization;

namespace PersonaEngine.Lib.ASR.Transcriber;

public static class WhisperNetSupportedLanguage
{
    private static readonly HashSet<string> supportedLanguages = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
        "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
        "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
        "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
        "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
        "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
        "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
        "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"
    ];

    public static bool IsSupported(CultureInfo cultureInfo) { return supportedLanguages.Contains(cultureInfo.TwoLetterISOLanguageName); }

    public static IEnumerable<CultureInfo> GetSupportedLanguages() { return supportedLanguages.Select(x => new CultureInfo(x)); }
}