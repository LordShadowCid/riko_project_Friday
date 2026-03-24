using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Applies spring bone forces when the window moves (e.g. during drag).
    /// Hair, clothes, and accessories sway dynamically in reaction to movement.
    /// Based on Mate-Engine AvatarGravityController.cs — VRM1 only.
    /// </summary>
    public class DragAnimationController : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        [Header("Settings")]
        [SerializeField] private float impactMultiplier = 0.05f;

        private Vector2Int _prevWindowPos;
        private readonly List<VRM10SpringBoneJoint> _joints = new();
        private Vrm10Instance _vrm;
        private IntPtr _hwnd;

        [StructLayout(LayoutKind.Sequential)]
        private struct RECT { public int left, top, right, bottom; }

        [DllImport("user32.dll")]
        private static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        [DllImport("user32.dll")]
        private static extern IntPtr GetActiveWindow();

        private void Start()
        {
            _hwnd = GetActiveWindow();
            _prevWindowPos = GetWindowPosition();
        }

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _joints.Clear();
            _joints.AddRange(vrm.GetComponentsInChildren<VRM10SpringBoneJoint>(true));
            Debug.Log($"[DragAnimation] Initialized with {_joints.Count} spring bone joints");
        }

        private void Update()
        {
            if (_hwnd == IntPtr.Zero || _joints.Count == 0 || _vrm == null) return;

            var pos = GetWindowPosition();
            var delta = pos - _prevWindowPos;
            _prevWindowPos = pos;

            if (delta == Vector2Int.zero) return;

            Vector3 force = new Vector3(-delta.x, delta.y, 0).normalized * impactMultiplier;

            foreach (var joint in _joints)
            {
                if (joint == null) continue;
                joint.m_gravityDir = force.normalized;
                joint.m_gravityPower = force.magnitude;
                _vrm.Runtime?.SpringBone?.SetJointLevel(joint.transform, joint.Blittable);
            }
        }

        private Vector2Int GetWindowPosition()
        {
            GetWindowRect(_hwnd, out RECT rect);
            return new Vector2Int(rect.left, rect.top);
        }
#else
        public void Initialize(Vrm10Instance vrm)
        {
            Debug.Log("[DragAnimation] Editor mode — spring bone forces disabled.");
        }
#endif
    }
}
