#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Annabeth.Editor
{
    public static class CommandLineBuild
    {
        public static void BuildWindows()
        {
            // Clean any corrupted FixAlpha features from the URP renderer data
            // before building. These were incorrectly injected by a previous
            // editor script and crash the URP pipeline at runtime.
            CleanRendererData();

            // CRITICAL: Preserve framebuffer alpha so the DWM compositor can
            // use per-pixel alpha for transparent window compositing.
            // Without this, Unity destroys the alpha channel in the backbuffer
            // and the character bleeds through / is see-through.
            PlayerSettings.preserveFramebufferAlpha = true;

            // Disable DXGI Flip Model for DWM per-pixel alpha compositing.
            // WS_EX_LAYERED + DwmExtendFrameIntoClientArea requires the legacy
            // BitBlt swap chain so DWM can read per-pixel alpha from the backbuffer.
            PlayerSettings.useFlipModelSwapchain = false;

            // Configure URP Pipeline Asset for transparent window alpha passthrough
            ConfigureURPAlpha();

            string buildPath = "Builds/AnnabethTest/Annabeth.exe";
            
            var scenes = new[] { "Assets/Scenes/SampleScene.unity" };
            
            // Verify scene exists
            foreach (var scene in scenes)
            {
                if (!System.IO.File.Exists(scene))
                {
                    Debug.LogError($"[Build] Scene not found: {scene}");
                    EditorApplication.Exit(1);
                    return;
                }
            }
            
            var options = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = buildPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };
            
            Debug.Log($"[Build] Building to: {buildPath}");
            var report = BuildPipeline.BuildPlayer(options);
            
            if (report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded)
            {
                Debug.Log($"[Build] SUCCESS - Size: {report.summary.totalSize / (1024*1024)} MB");
                EditorApplication.Exit(0);
            }
            else
            {
                Debug.LogError($"[Build] FAILED: {report.summary.result}");
                foreach (var step in report.steps)
                {
                    foreach (var msg in step.messages)
                    {
                        if (msg.type == LogType.Error)
                            Debug.LogError($"[Build] {msg.content}");
                    }
                }
                EditorApplication.Exit(1);
            }
        }

        /// <summary>
        /// Configure URP Pipeline Asset for transparent window rendering:
        /// 1. Disable HDR (forces intermediate RT that clobbers alpha)
        /// 2. Enable AlphaProcessing (Unity 6000 requirement for alpha preservation)
        /// From Kirurobo/UniWindowController issue #42:
        /// "Unity6000系の場合、HDRを無効にするだけだと背景が透過しないようです。
        ///  AlphaProcessingを有効化することで透過するようです"
        /// </summary>
        static void ConfigureURPAlpha()
        {
            var rpAsset = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
            if (rpAsset == null) return;

            var so = new SerializedObject(rpAsset);
            bool changed = false;

            // Disable HDR on pipeline asset — HDR forces intermediate RT that kills alpha
            var hdrProp = so.FindProperty("m_SupportsHDR");
            if (hdrProp != null && hdrProp.boolValue)
            {
                Debug.Log("[Build] Disabling HDR on URP Pipeline Asset (required for alpha passthrough)");
                hdrProp.boolValue = false;
                changed = true;
            }

            // Enable AllowPostProcessAlphaOutput — REQUIRED for Unity 6000.
            // URP 17+ always uses an intermediate render target, then blits to
            // the backbuffer. Without this flag, URP's blit shader writes alpha=1
            // everywhere, making the window fully opaque black.
            // With it enabled, URP preserves per-pixel alpha through the blit.
            // Reference: Kirurobo/UniWindowController issue #42:
            // "Unity6000系 — AlphaProcessingを有効化することで透過するようです"
            var alphaProp = so.FindProperty("m_AllowPostProcessAlphaOutput");
            if (alphaProp != null && !alphaProp.boolValue)
            {
                Debug.Log("[Build] Enabling AllowPostProcessAlphaOutput (Unity 6000 requirement for alpha passthrough)");
                alphaProp.boolValue = true;
                changed = true;
            }

            if (changed)
            {
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty(rpAsset);
                AssetDatabase.SaveAssets();
                Debug.Log("[Build] URP Pipeline Asset configured for transparent window alpha.");
            }
        }

        /// <summary>
        /// Clean URP Renderer Data for transparent window rendering:
        /// 1. Remove any FixAlphaRendererFeature instances (corrupted sub-assets)
        /// 2. Remove SSAO (forces intermediate render target, destroys alpha)
        /// 3. Force Forward rendering mode (Deferred uses G-buffers → kills alpha)
        /// </summary>
        static void CleanRendererData()
        {
            var rpAsset = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
            if (rpAsset == null) return;

            var field = typeof(UniversalRenderPipelineAsset)
                .GetField("m_RendererDataList",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field == null) return;

            var dataList = field.GetValue(rpAsset) as ScriptableRendererData[];
            if (dataList == null) return;

            foreach (var rendererData in dataList)
            {
                if (rendererData == null) continue;

                var so = new SerializedObject(rendererData);
                var featuresProp = so.FindProperty("m_RendererFeatures");
                var mapProp = so.FindProperty("m_RendererFeatureMap");

                // Force Forward rendering (Deferred = 2, Forward = 0)
                // Deferred uses G-buffers + intermediate RTs that clobber alpha channel.
                var renderModeProp = so.FindProperty("m_RenderingMode");
                bool changed = false;

                if (renderModeProp != null && renderModeProp.intValue != 0)
                {
                    Debug.Log($"[Build] Switching {rendererData.name} from RenderingMode {renderModeProp.intValue} to Forward (0)");
                    renderModeProp.intValue = 0; // Forward
                    changed = true;
                }

                if (featuresProp == null) { if (changed) { so.ApplyModifiedPropertiesWithoutUndo(); EditorUtility.SetDirty(rendererData); } continue; }

                for (int i = featuresProp.arraySize - 1; i >= 0; i--)
                {
                    var obj = featuresProp.GetArrayElementAtIndex(i).objectReferenceValue;

                    // Remove null entries, FixAlpha features, and SSAO
                    // SSAO forces depth/normals pre-pass + intermediate RT which
                    // destroys the alpha channel needed for transparent window.
                    bool shouldRemove = (obj == null) ||
                        obj.GetType().Name.Contains("FixAlpha") ||
                        obj.GetType().Name.Contains("ScreenSpaceAmbientOcclusion");

                    if (shouldRemove)
                    {
                        string featureName = obj != null ? obj.GetType().Name : "null";
                        Debug.Log($"[Build] Removing renderer feature: {featureName}");

                        if (obj != null)
                        {
                            string assetPath = AssetDatabase.GetAssetPath(obj);
                            if (!string.IsNullOrEmpty(assetPath) && AssetDatabase.IsSubAsset(obj))
                                AssetDatabase.RemoveObjectFromAsset(obj);
                            Object.DestroyImmediate(obj, true);
                        }

                        // Clear object ref first (required), then delete element
                        featuresProp.GetArrayElementAtIndex(i).objectReferenceValue = null;
                        featuresProp.DeleteArrayElementAtIndex(i);

                        if (mapProp != null && i < mapProp.arraySize)
                            mapProp.DeleteArrayElementAtIndex(i);

                        changed = true;
                    }
                }

                if (changed)
                {
                    so.ApplyModifiedPropertiesWithoutUndo();
                    EditorUtility.SetDirty(rendererData);
                    AssetDatabase.SaveAssets();
                    Debug.Log($"[Build] Cleaned renderer data: {rendererData.name}");
                }
            }
        }
    }
}
#endif
