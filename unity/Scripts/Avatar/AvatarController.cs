using System.Threading.Tasks;
using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    public class AvatarController : MonoBehaviour
    {
        [Header("VRM Settings")]
        [SerializeField] private string vrmPath = "Models/claire_avatar.vrm";
        [SerializeField] private bool loadOnStart = true;
        
        [Header("Components")]
        [SerializeField] private LipSyncController lipSyncController;
        [SerializeField] private EmotionController emotionController;
        [SerializeField] private BlinkController blinkController;
        [SerializeField] private EyeTrackingController eyeTrackingController;

        private Vrm10Instance _vrmInstance;
        private Vrm10RuntimeExpression _expression;
        private bool _isLoaded;

        // Material opacity tracking — stores material, its color property name, and cached alpha
        private readonly System.Collections.Generic.List<(Material mat, string colorProp, float origAlpha)> _materialAlphas
            = new System.Collections.Generic.List<(Material, string, float)>();

        public Vrm10Instance VrmInstance => _vrmInstance;
        public Vrm10RuntimeExpression Expression => _expression;
        public bool IsLoaded => _isLoaded;

        public event System.Action<Vrm10Instance> OnVrmLoaded;

        private async void Start()
        {
            if (loadOnStart)
            {
                // Check for user-saved model path first
                string savedPath = Core.SettingsManager.Instance?.data.selectedModelPath;
                if (!string.IsNullOrEmpty(savedPath) && System.IO.File.Exists(savedPath))
                    await LoadVRM(savedPath, isAbsolutePath: true);
                else
                    await LoadVRM(vrmPath);
            }
        }

        public async Task LoadVRM(string path, bool isAbsolutePath = false)
        {
            if (_vrmInstance != null)
            {
                Destroy(_vrmInstance.gameObject);
                _vrmInstance = null;
                _expression = null;
                _isLoaded = false;
                // Release resources from previous model (VRM assets can be large)
                Resources.UnloadUnusedAssets();
            }

            try
            {
                string fullPath = isAbsolutePath
                    ? path
                    : System.IO.Path.Combine(Application.streamingAssetsPath, path);
                Debug.Log($"[AvatarController] Loading VRM from: {fullPath}");

                if (!System.IO.File.Exists(fullPath))
                {
                    Debug.LogError($"[AvatarController] File not found: {fullPath}");
                    return;
                }

                var sw = System.Diagnostics.Stopwatch.StartNew();
                
                _vrmInstance = await Vrm10.LoadPathAsync(fullPath,
                    canLoadVrm0X: true,
                    showMeshes: true,
                    materialGenerator: new UrpVrm10MaterialDescriptorGenerator(),
                    ct: destroyCancellationToken);

                sw.Stop();
                Debug.Log($"[AvatarController] LoadPathAsync returned in {sw.ElapsedMilliseconds}ms, result={((_vrmInstance != null) ? _vrmInstance.gameObject.name : "NULL")}");

                if (_vrmInstance == null)
                {
                    Debug.LogError("[AvatarController] Failed to load VRM - returned null!");
                    return;
                }

                _vrmInstance.transform.SetParent(transform, false);
                _vrmInstance.transform.localPosition = Vector3.zero;
                _vrmInstance.transform.localRotation = Quaternion.identity;

                // VRM 1.0 models face +Z natively. Place camera on +Z side
                // looking back toward origin. Framed for full body visibility.
                var cam = Camera.main;
                if (cam != null)
                {
                    cam.transform.position = new Vector3(0f, 0.85f, 2.6f);
                    cam.transform.rotation = Quaternion.Euler(0f, 180f, 0f);
                    cam.fieldOfView = 35f;
                    cam.nearClipPlane = 0.1f;
                }

                _expression = _vrmInstance.Runtime.Expression;

                InitializeControllers();
                ApplyRelaxedPose();
                CacheMaterialAlphas();

                _isLoaded = true;
                Debug.Log("[AvatarController] VRM loaded successfully!");
                OnVrmLoaded?.Invoke(_vrmInstance);
            }
            catch (System.OperationCanceledException)
            {
                Debug.LogWarning("[AvatarController] VRM loading was cancelled");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[AvatarController] Error loading VRM: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
            }
        }

        private void InitializeControllers()
        {
            if (lipSyncController == null)
                lipSyncController = GetComponentInChildren<LipSyncController>();
            if (emotionController == null)
                emotionController = GetComponentInChildren<EmotionController>();
            if (blinkController == null)
                blinkController = GetComponentInChildren<BlinkController>();
            if (eyeTrackingController == null)
                eyeTrackingController = GetComponentInChildren<EyeTrackingController>();

            lipSyncController?.Initialize(_vrmInstance);
            emotionController?.Initialize(_vrmInstance);
            blinkController?.Initialize(_vrmInstance);
            eyeTrackingController?.Initialize(_vrmInstance);
        }

        public Transform GetBone(HumanBodyBones bone)
        {
            if (!_isLoaded || _vrmInstance == null) return null;
            
            var animator = _vrmInstance.GetComponent<Animator>();
            return animator?.GetBoneTransform(bone);
        }

        /// <summary>
        /// Sets a natural resting pose via the VRM10 ControlRig so the avatar
        /// doesn't remain in T-pose when no animation clip is playing.
        /// </summary>
        private void ApplyRelaxedPose()
        {
            var rig = _vrmInstance.Runtime?.ControlRig;
            if (rig == null) return;

            var lUA = rig.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            var rUA = rig.GetBoneTransform(HumanBodyBones.RightUpperArm);
            var lLA = rig.GetBoneTransform(HumanBodyBones.LeftLowerArm);
            var rLA = rig.GetBoneTransform(HumanBodyBones.RightLowerArm);

            if (lUA != null) lUA.localRotation = Quaternion.Euler(0f, 0f, 72f);
            if (rUA != null) rUA.localRotation = Quaternion.Euler(0f, 0f, -72f);
            if (lLA != null) lLA.localRotation = Quaternion.Euler(0f, -35f, 0f);
            if (rLA != null) rLA.localRotation = Quaternion.Euler(0f, 35f, 0f);
        }

        /// <summary>
        /// Cache original material alpha values after VRM load.
        /// Checks both _Color (standard) and _BaseColor (URP) properties.
        /// Forces all materials to fully opaque rendering.
        /// </summary>
        private void CacheMaterialAlphas()
        {
            _materialAlphas.Clear();
            if (_vrmInstance == null) return;

            foreach (var r in _vrmInstance.GetComponentsInChildren<Renderer>())
            {
                foreach (var mat in r.materials)
                {
                    if (mat == null) continue;

                    // Determine which color property this material uses
                    string colorProp = null;
                    if (mat.HasProperty("_Color")) colorProp = "_Color";
                    else if (mat.HasProperty("_BaseColor")) colorProp = "_BaseColor";

                    float origAlpha = 1f;
                    if (colorProp != null)
                    {
                        origAlpha = mat.GetColor(colorProp).a;
                        // Force alpha to 1.0 so character is fully opaque
                        var c = mat.GetColor(colorProp);
                        c.a = 1f;
                        mat.SetColor(colorProp, c);
                        origAlpha = 1f;
                    }

                    // Force shade color alpha too (MToon shaders)
                    if (mat.HasProperty("_ShadeColor"))
                    {
                        var sc = mat.GetColor("_ShadeColor");
                        sc.a = 1f;
                        mat.SetColor("_ShadeColor", sc);
                    }

                    // Force opaque rendering mode
                    ForceOpaqueRendering(mat);

                    _materialAlphas.Add((mat, colorProp ?? "_Color", origAlpha));
                }
            }
            Debug.Log($"[AvatarController] Cached {_materialAlphas.Count} materials, forced opaque rendering.");
        }

        /// <summary>
        /// Force a material to render as opaque by setting surface type,
        /// render queue, ZWrite, blend modes, and disabling transparency keywords.
        /// Aggressively overrides all known transparency properties for MToon10/URP.
        /// </summary>
        private void ForceOpaqueRendering(Material mat)
        {
            // MToon10 alpha mode: 0=Opaque
            if (mat.HasProperty("_M_AlphaMode"))
                mat.SetFloat("_M_AlphaMode", 0f);

            // MToon10 transparent-with-ZWrite flag
            if (mat.HasProperty("_M_TransparentWithZWrite"))
                mat.SetFloat("_M_TransparentWithZWrite", 0f);

            // URP Lit surface type: 0=Opaque
            if (mat.HasProperty("_Surface"))
                mat.SetFloat("_Surface", 0f);

            // Force ZWrite on (required for opaque)
            if (mat.HasProperty("_ZWrite"))
                mat.SetFloat("_ZWrite", 1f);

            // Force GPU blend mode to fully opaque (One, Zero)
            // This overrides any shader-level alpha blending
            if (mat.HasProperty("_SrcBlend"))
                mat.SetFloat("_SrcBlend", 1f); // BlendMode.One
            if (mat.HasProperty("_DstBlend"))
                mat.SetFloat("_DstBlend", 0f); // BlendMode.Zero
            if (mat.HasProperty("_SrcBlendAlpha"))
                mat.SetFloat("_SrcBlendAlpha", 1f);
            if (mat.HasProperty("_DstBlendAlpha"))
                mat.SetFloat("_DstBlendAlpha", 0f);

            // Disable alpha cutout
            if (mat.HasProperty("_AlphaClip"))
                mat.SetFloat("_AlphaClip", 0f);
            if (mat.HasProperty("_M_CutoutThresholdValue"))
                mat.SetFloat("_M_CutoutThresholdValue", 0f);

            // Disable all transparency blend keywords
            mat.DisableKeyword("_ALPHABLEND_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.DisableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.DisableKeyword("_ALPHAMODULATE_ON");

            // Enable dark outline so character stands out against any wallpaper.
            // MToon10 outline: ScreenCoordinates mode with dark color.
            if (mat.HasProperty("_M_OutlineWidthMode"))
                mat.SetFloat("_M_OutlineWidthMode", 2f); // ScreenCoordinates
            if (mat.HasProperty("_M_OutlineWidth"))
                mat.SetFloat("_M_OutlineWidth", 0.08f); // thin but visible
            if (mat.HasProperty("_M_OutlineColor"))
                mat.SetColor("_M_OutlineColor", new Color(0.05f, 0.05f, 0.08f, 1f)); // near-black
            if (mat.HasProperty("_M_OutlineLightingMixFactor"))
                mat.SetFloat("_M_OutlineLightingMixFactor", 0f); // pure unlit outline color
            mat.EnableKeyword("_MTOON_OUTLINE_ON");

            // Set render queue to Geometry (opaque = 2000)
            mat.renderQueue = 2000;
        }

        /// <summary>
        /// Per-frame enforcement: re-force opaque on all VRM materials.
        /// UniVRM's runtime may reset material properties each frame during
        /// expression evaluation. This ensures they stay opaque.
        /// </summary>
        private void LateUpdate()
        {
            if (!_isLoaded || _materialAlphas.Count == 0) return;

            foreach (var (mat, colorProp, _) in _materialAlphas)
            {
                if (mat == null) continue;

                // Re-force color alpha to 1
                if (mat.HasProperty(colorProp))
                {
                    var c = mat.GetColor(colorProp);
                    if (c.a < 0.99f)
                    {
                        c.a = 1f;
                        mat.SetColor(colorProp, c);
                    }
                }

                // Re-force shade color alpha (MToon)
                if (mat.HasProperty("_ShadeColor"))
                {
                    var sc = mat.GetColor("_ShadeColor");
                    if (sc.a < 0.99f)
                    {
                        sc.a = 1f;
                        mat.SetColor("_ShadeColor", sc);
                    }
                }

                // Re-force opaque blend modes every frame (UniVRM may reset these)
                if (mat.HasProperty("_SrcBlend"))
                {
                    float src = mat.GetFloat("_SrcBlend");
                    if (src != 1f)
                    {
                        mat.SetFloat("_SrcBlend", 1f);
                        mat.SetFloat("_DstBlend", 0f);
                    }
                }
                if (mat.HasProperty("_SrcBlendAlpha"))
                {
                    float srcA = mat.GetFloat("_SrcBlendAlpha");
                    if (srcA != 1f)
                    {
                        mat.SetFloat("_SrcBlendAlpha", 1f);
                        mat.SetFloat("_DstBlendAlpha", 0f);
                    }
                }
            }
        }

        /// <summary>
        /// Apply character opacity by multiplying all material alpha values.
        /// 1.0 = fully opaque (default), 0.1 = very transparent.
        /// </summary>
        public void ApplyCharacterOpacity(float opacity)
        {
            opacity = Mathf.Clamp(opacity, 0.1f, 1f);
            foreach (var (mat, colorProp, origAlpha) in _materialAlphas)
            {
                if (mat == null || !mat.HasProperty(colorProp)) continue;
                var c = mat.GetColor(colorProp);
                c.a = origAlpha * opacity;
                mat.SetColor(colorProp, c);
            }
        }

        /// <summary>
        /// Apply avatar scale from settings.
        /// </summary>
        public void ApplyAvatarSize(float size)
        {
            if (_vrmInstance == null) return;
            size = Mathf.Clamp(size, 0.5f, 2f);
            _vrmInstance.transform.localScale = Vector3.one * size;
        }

        private void OnDestroy()
        {
            if (_vrmInstance != null)
            {
                Destroy(_vrmInstance.gameObject);
            }
        }
    }
}