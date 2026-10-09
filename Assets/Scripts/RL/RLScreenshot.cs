using System.IO;
using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// -rlScreenshot file.png [-rlScreenshotSeconds 3]: saves the screen (UI included) after a few
    /// seconds, then quits - for documentation, e.g. a picture of the TRAINING STAGES screen:
    ///
    ///   NavalTrainer.x86_64 -rlStages -rlScreenshot stages.png
    /// </summary>
    public class RLScreenshot : MonoBehaviour
    {
        string _path;
        float _after;
        bool _taken;
        float _quitAt;

        public static RLScreenshot Create(Transform parent)
        {
            var go = new GameObject("RLScreenshot");
            go.transform.SetParent(parent, false);
            var s = go.AddComponent<RLScreenshot>();
            s._path = Path.GetFullPath(RLCommandLine.Str("-rlScreenshot", "screenshot.png"));
            s._after = RLCommandLine.Int("-rlScreenshotSeconds", 3);
            return s;
        }

        void Update()
        {
            if (!_taken && Time.realtimeSinceStartup >= _after)
            {
                ScreenCapture.CaptureScreenshot(_path);
                Debug.Log("[RL] screenshot: " + _path);
                _taken = true;
                _quitAt = Time.realtimeSinceStartup + 1f;          // the file is written at the end of the frame
            }
            else if (_taken && Time.realtimeSinceStartup >= _quitAt) Application.Quit();
        }
    }
}
