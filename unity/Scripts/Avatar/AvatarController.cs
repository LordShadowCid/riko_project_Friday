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
                // looking back toward origin with Y-rotation only.
                // NO Z-roll — there is no URP Y-flip to compensate for.
                var cam = Camera.main;
                if (cam != null)
                {
                    cam.transform.position = new Vector3(0f, 1.3f, 3.0f);
                    cam.transform.rotation = Quaternion.Euler(0f, 180f, 0f);
                }

                _expression = _vrmInstance.Runtime.Expression;

                InitializeControllers();
                ApplyRelaxedPose();

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

            if (lUA != null) lUA.localRotation = Quaternion.Euler(0f, 0f, 55f);
            if (rUA != null) rUA.localRotation = Quaternion.Euler(0f, 0f, -55f);
            if (lLA != null) lLA.localRotation = Quaternion.Euler(0f, -20f, 0f);
            if (rLA != null) rLA.localRotation = Quaternion.Euler(0f, 20f, 0f);
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