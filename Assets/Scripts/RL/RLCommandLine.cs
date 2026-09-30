using System;

namespace Naval.RL
{
    /// <summary>Command-line switches for headless training players: -rlTrain -rlPort 5005.</summary>
    public static class RLCommandLine
    {
        public static bool Has(string flag)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
                if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }

        public static string Str(string flag, string fallback)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase) && !args[i + 1].StartsWith("-"))
                    return args[i + 1];
            return fallback;
        }

        public static int Int(string flag, int fallback)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase) && int.TryParse(args[i + 1], out int v))
                    return v;
            return fallback;
        }
    }
}
