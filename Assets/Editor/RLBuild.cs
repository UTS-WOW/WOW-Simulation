using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Naval.RL.EditorTools
{
    /// <summary>
    /// Builds the player the trainer launches in parallel (Training/train.py --unity-binary ...).
    /// The build is an ordinary Linux player; run it with -batchmode -nographics -rlTrain -rlPort N.
    /// Command line:  Unity -batchmode -projectPath . -executeMethod Naval.RL.EditorTools.RLBuild.BuildLinuxTrainer
    /// </summary>
    public static class RLBuild
    {
        public const string OutputPath = "Builds/NavalTrainer/NavalTrainer.x86_64";

        [MenuItem("Naval/RL/Build Training Player (Linux)")]
        public static void BuildLinuxTrainer()
        {
            var scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
            // -rlBuildPath lets a build made from a copy of the project land in the real one
            string output = OutputPath;
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == "-rlBuildPath") output = args[i + 1];
            var options = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = output,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None
            };
            var report = BuildPipeline.BuildPlayer(options);
            bool ok = report.summary.result == BuildResult.Succeeded;
            Debug.Log("[RL] training player build " + (ok ? "succeeded: " + output : "FAILED: " + report.summary.result));
            if (Application.isBatchMode) EditorApplication.Exit(ok ? 0 : 1);
        }
    }
}
