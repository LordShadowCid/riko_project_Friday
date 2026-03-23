using UnityEngine;
using UnityEngine.InputSystem;
using UniVRM10;

namespace Annabeth.Interaction
{
    /// <summary>
    /// Detects clicks on the avatar and triggers reactions.
    /// Reactions: surprised, happy, annoyed (random). Uses raycasting against
    /// the VRM's SkinnedMeshRenderers. Head and body zones trigger different
    /// reactions.
    /// </summary>
    public class TouchReactionController : MonoBehaviour
    {
        [Header("Settings")]
        [SerializeField] private float reactionDuration = 1.5f;
        [SerializeField] private float cooldown = 0.8f;
        [SerializeField] private float expressionBlendSpeed = 8f;

        private Vrm10Instance _vrm;
        private Vrm10RuntimeExpression _expression;
        private Camera _cam;

        private float _reactionTimer;
        private float _cooldownTimer;
        private ExpressionKey _activeReaction;
        private float _reactionWeight;
        private bool _reacting;

        // Head zone: Y above shoulders (local space threshold)
        private float _headYThreshold;

        private static readonly ExpressionKey[] HeadReactions = new[]
        {
            ExpressionKey.Surprised,
            ExpressionKey.Happy,
            ExpressionKey.Angry,
        };

        private static readonly ExpressionKey[] BodyReactions = new[]
        {
            ExpressionKey.Happy,
            ExpressionKey.Surprised,
        };

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
            _expression = vrm?.Runtime?.Expression;
            _cam = Camera.main;

            // Estimate head zone: top 25% of model height
            var renderers = vrm.GetComponentsInChildren<SkinnedMeshRenderer>();
            float maxY = float.MinValue;
            float minY = float.MaxValue;
            foreach (var r in renderers)
            {
                var bounds = r.bounds;
                if (bounds.max.y > maxY) maxY = bounds.max.y;
                if (bounds.min.y < minY) minY = bounds.min.y;
            }
            _headYThreshold = Mathf.Lerp(minY, maxY, 0.75f);
        }

        private void Update()
        {
            if (_expression == null || _cam == null) return;

            // Cooldown
            if (_cooldownTimer > 0f)
            {
                _cooldownTimer -= Time.deltaTime;
            }

            // Handle active reaction fade
            if (_reacting)
            {
                _reactionTimer -= Time.deltaTime;

                if (_reactionTimer > 0f)
                {
                    // Blend in
                    _reactionWeight = Mathf.MoveTowards(_reactionWeight, 1f, Time.deltaTime * expressionBlendSpeed);
                }
                else
                {
                    // Blend out
                    _reactionWeight = Mathf.MoveTowards(_reactionWeight, 0f, Time.deltaTime * expressionBlendSpeed);
                    if (_reactionWeight <= 0.01f)
                    {
                        _expression.SetWeight(_activeReaction, 0f);
                        _reacting = false;
                    }
                }

                if (_reacting)
                {
                    _expression.SetWeight(_activeReaction, _reactionWeight);
                }
            }

            // Detect click (InputSystem)
            var mouse = Mouse.current;
            if (mouse != null && mouse.leftButton.wasPressedThisFrame && _cooldownTimer <= 0f && !_reacting)
            {
                TryTouch();
            }
        }

        private void TryTouch()
        {
            var mousePos = Mouse.current?.position.ReadValue() ?? Vector2.zero;
            Ray ray = _cam.ScreenPointToRay(mousePos);

            // Raycast against all colliders/mesh renderers in the VRM hierarchy
            if (_vrm == null) return;

            var renderers = _vrm.GetComponentsInChildren<SkinnedMeshRenderer>();
            float closestDist = float.MaxValue;
            Vector3 hitPoint = Vector3.zero;
            bool hit = false;

            foreach (var r in renderers)
            {
                if (r.bounds.IntersectRay(ray, out float dist) && dist < closestDist)
                {
                    closestDist = dist;
                    hitPoint = ray.GetPoint(dist);
                    hit = true;
                }
            }

            if (!hit) return;

            // Determine zone
            bool isHead = hitPoint.y >= _headYThreshold;
            var reactions = isHead ? HeadReactions : BodyReactions;
            _activeReaction = reactions[Random.Range(0, reactions.Length)];

            _reacting = true;
            _reactionTimer = reactionDuration;
            _reactionWeight = 0f;
            _cooldownTimer = cooldown;

            string zone = isHead ? "head" : "body";
            Debug.Log($"[TouchReaction] Touched {zone} -> {_activeReaction}");
        }
    }
}
