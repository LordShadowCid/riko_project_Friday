using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UniVRM10;

namespace Annabeth.Interaction
{
    /// <summary>
    /// Body zone detected from closest-bone proximity after raycast hit.
    /// </summary>
    public enum TouchZone
    {
        Head,
        Chest,
        Belly,
        LeftArm,
        RightArm,
        LeftHand,
        RightHand,
        Legs,
    }

    /// <summary>
    /// Detects clicks on the avatar and triggers zone-specific reactions.
    /// Uses raycasting against SkinnedMeshRenderers, then finds the closest
    /// humanoid bone to determine which body zone was touched. Each zone has
    /// its own set of expression reactions.
    /// </summary>
    public class TouchReactionController : MonoBehaviour
    {
        /// <summary>Fired on touch reaction with (hitWorldPos, zone).</summary>
        public event System.Action<Vector3, TouchZone> OnTouchReaction;

        [Header("Settings")]
        [SerializeField] private float reactionDuration = 1.5f;
        [SerializeField] private float cooldown = 0.8f;
        [SerializeField] private float expressionBlendSpeed = 8f;

        private Vrm10Instance _vrm;
        private Vrm10RuntimeExpression _expression;
        private Camera _cam;
        private SkinnedMeshRenderer[] _renderers;

        private float _reactionTimer;
        private float _cooldownTimer;
        private ExpressionKey _activeReaction;
        private float _reactionWeight;
        private bool _reacting;

        // Bone → zone mapping built on Initialize
        private readonly List<(Transform bone, TouchZone zone)> _boneZones = new();

        // Per-zone reaction tables
        private static readonly Dictionary<TouchZone, ExpressionKey[]> ZoneReactions = new()
        {
            { TouchZone.Head,      new[] { ExpressionKey.Happy, ExpressionKey.Surprised, ExpressionKey.Relaxed } },
            { TouchZone.Chest,     new[] { ExpressionKey.Surprised, ExpressionKey.Angry } },
            { TouchZone.Belly,     new[] { ExpressionKey.Happy, ExpressionKey.Surprised } },
            { TouchZone.LeftArm,   new[] { ExpressionKey.Happy, ExpressionKey.Surprised } },
            { TouchZone.RightArm,  new[] { ExpressionKey.Happy, ExpressionKey.Surprised } },
            { TouchZone.LeftHand,  new[] { ExpressionKey.Happy, ExpressionKey.Relaxed } },
            { TouchZone.RightHand, new[] { ExpressionKey.Happy, ExpressionKey.Relaxed } },
            { TouchZone.Legs,      new[] { ExpressionKey.Surprised, ExpressionKey.Angry } },
        };

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _expression = vrm?.Runtime?.Expression;
            _cam = Camera.main;
            _renderers = vrm.GetComponentsInChildren<SkinnedMeshRenderer>();

            BuildBoneZoneMap(vrm);
        }

        private void BuildBoneZoneMap(Vrm10Instance vrm)
        {
            _boneZones.Clear();
            var animator = vrm.GetComponentInChildren<Animator>();
            if (animator == null) return;

            // Map humanoid bones to zones (order doesn't matter — closest wins)
            var mapping = new (HumanBodyBones bone, TouchZone zone)[]
            {
                (HumanBodyBones.Head,          TouchZone.Head),
                (HumanBodyBones.Neck,          TouchZone.Head),
                (HumanBodyBones.UpperChest,    TouchZone.Chest),
                (HumanBodyBones.Chest,         TouchZone.Chest),
                (HumanBodyBones.Spine,         TouchZone.Belly),
                (HumanBodyBones.Hips,          TouchZone.Belly),
                (HumanBodyBones.LeftUpperArm,  TouchZone.LeftArm),
                (HumanBodyBones.LeftLowerArm,  TouchZone.LeftArm),
                (HumanBodyBones.RightUpperArm, TouchZone.RightArm),
                (HumanBodyBones.RightLowerArm, TouchZone.RightArm),
                (HumanBodyBones.LeftHand,      TouchZone.LeftHand),
                (HumanBodyBones.RightHand,     TouchZone.RightHand),
                (HumanBodyBones.LeftUpperLeg,  TouchZone.Legs),
                (HumanBodyBones.RightUpperLeg, TouchZone.Legs),
                (HumanBodyBones.LeftLowerLeg,  TouchZone.Legs),
                (HumanBodyBones.RightLowerLeg, TouchZone.Legs),
                (HumanBodyBones.LeftFoot,      TouchZone.Legs),
                (HumanBodyBones.RightFoot,     TouchZone.Legs),
            };

            foreach (var (bone, zone) in mapping)
            {
                var t = animator.GetBoneTransform(bone);
                if (t != null)
                    _boneZones.Add((t, zone));
            }
        }

        private TouchZone GetZoneFromHit(Vector3 hitPoint)
        {
            if (_boneZones.Count == 0) return TouchZone.Chest; // fallback

            float bestDist = float.MaxValue;
            TouchZone bestZone = TouchZone.Chest;

            foreach (var (bone, zone) in _boneZones)
            {
                float d = (bone.position - hitPoint).sqrMagnitude;
                if (d < bestDist)
                {
                    bestDist = d;
                    bestZone = zone;
                }
            }

            return bestZone;
        }

        private void Update()
        {
            if (_expression == null || _cam == null) return;

            if (_cooldownTimer > 0f)
                _cooldownTimer -= Time.deltaTime;

            // Handle active reaction fade
            if (_reacting)
            {
                _reactionTimer -= Time.deltaTime;
                float target = _reactionTimer > 0f ? 1f : 0f;
                _reactionWeight = Mathf.MoveTowards(_reactionWeight, target, Time.deltaTime * expressionBlendSpeed);

                if (target == 0f && _reactionWeight <= 0.01f)
                {
                    _expression.SetWeight(_activeReaction, 0f);
                    _reacting = false;
                }
                else
                {
                    _expression.SetWeight(_activeReaction, _reactionWeight);
                }
            }

            // Detect click
            var mouse = Mouse.current;
            if (mouse != null && mouse.leftButton.wasPressedThisFrame && _cooldownTimer <= 0f && !_reacting)
                TryTouch();
        }

        private void TryTouch()
        {
            var mousePos = Mouse.current?.position.ReadValue() ?? Vector2.zero;
            Ray ray = _cam.ScreenPointToRay(mousePos);

            if (_vrm == null || _renderers == null) return;

            float closestDist = float.MaxValue;
            Vector3 hitPoint = Vector3.zero;
            bool hit = false;

            foreach (var r in _renderers)
            {
                if (r.bounds.IntersectRay(ray, out float dist) && dist < closestDist)
                {
                    closestDist = dist;
                    hitPoint = ray.GetPoint(dist);
                    hit = true;
                }
            }

            if (!hit) return;

            TouchZone zone = GetZoneFromHit(hitPoint);

            if (!ZoneReactions.TryGetValue(zone, out var reactions))
                reactions = new[] { ExpressionKey.Surprised };

            _activeReaction = reactions[UnityEngine.Random.Range(0, reactions.Length)];
            _reacting = true;
            _reactionTimer = reactionDuration;
            _reactionWeight = 0f;
            _cooldownTimer = cooldown;

            OnTouchReaction?.Invoke(hitPoint, zone);
            Debug.Log($"[TouchReaction] Touched {zone} -> {_activeReaction}");
        }
    }
}
