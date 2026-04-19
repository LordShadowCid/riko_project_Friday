#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace Annabeth.Editor
{
    /// <summary>
    /// Auto-adds VRM/MToon10 and URP shaders to "Always Included Shaders"
    /// so they survive build stripping and are available for runtime VRM loading.
    /// Runs automatically when Unity recompiles scripts.
    /// </summary>
    [InitializeOnLoad]
    public static class VrmShaderIncluder
    {
        private static readonly string[] RequiredShaderNames =
        {
            "VRM10/Universal Render Pipeline/MToon10",
            "Universal Render Pipeline/Lit",
            "Universal Render Pipeline/Unlit",
            "Hidden/Annabeth/FixAlpha",
        };

        static VrmShaderIncluder()
        {
            EnsureShadersIncluded();
        }

        private static void EnsureShadersIncluded()
        {
            var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset");
            if (assets == null || assets.Length == 0) return;

            var graphicsSettings = new SerializedObject(assets[0]);
            var alwaysIncluded = graphicsSettings.FindProperty("m_AlwaysIncludedShaders");
            if (alwaysIncluded == null) return;

            bool changed = false;

            foreach (var shaderName in RequiredShaderNames)
            {
                var shader = Shader.Find(shaderName);
                if (shader == null)
                {
                    Debug.LogWarning($"[VrmShaderIncluder] Shader not found: {shaderName}");
                    continue;
                }

                bool found = false;
                for (int i = 0; i < alwaysIncluded.arraySize; i++)
                {
                    if (alwaysIncluded.GetArrayElementAtIndex(i).objectReferenceValue == shader)
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    alwaysIncluded.InsertArrayElementAtIndex(alwaysIncluded.arraySize);
                    alwaysIncluded.GetArrayElementAtIndex(alwaysIncluded.arraySize - 1)
                        .objectReferenceValue = shader;
                    changed = true;
                    Debug.Log($"[VrmShaderIncluder] Added '{shaderName}' to Always Included Shaders");
                }
            }

            if (changed)
            {
                graphicsSettings.ApplyModifiedProperties();
                Debug.Log("[VrmShaderIncluder] Graphics settings updated — VRM shaders will be included in builds");
            }
        }
    }
}
#endif
