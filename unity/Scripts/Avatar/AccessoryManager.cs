using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Bone-tracked accessories system based on Mate Engine's AccessoiresHandler.
    /// Loads OBJ/prefab accessories, parents them to avatar bones with smoothness lerp.
    /// Persists equipped state per VRM model to JSON.
    /// </summary>
    public class AccessoryManager : MonoBehaviour
    {
        /// <summary>Fired when any accessory is toggled on/off.</summary>
        public event Action<string, bool> OnAccessoryToggled;

        // ── State ───────────────────────────────────────────────
        private Animator _animator;
        private readonly List<AccessoryInstance> _active = new List<AccessoryInstance>();
        private readonly Dictionary<string, AccessoryDefinition> _catalog = new Dictionary<string, AccessoryDefinition>();

        [SerializeField] private float positionSmooth = 12f;
        [SerializeField] private float rotationSmooth = 12f;

        private string _currentModelId = "";
        private bool _stateDirty;
        private static bool _catalogScanned;
        private static readonly Dictionary<string, AccessoryDefinition> _cachedCatalog
            = new Dictionary<string, AccessoryDefinition>();

        // ── Persistence path ────────────────────────────────────
        private string StatePath => Path.Combine(Application.persistentDataPath, "accessory_state.json");

        // ── Public API ──────────────────────────────────────────

        /// <summary>Call after VRM loads to cache the animator and rebuild accessories.</summary>
        public void Initialize(UniVRM10.Vrm10Instance vrm)
        {
            // Tear down any existing accessories
            RemoveAll();

            _animator = vrm.GetComponentInChildren<Animator>();
            if (_animator == null)
            {
                Debug.LogWarning("[AccessoryManager] VRM has no Animator, accessories disabled");
                return;
            }
            _currentModelId = vrm.name;

            // Scan catalog folder (once across model swaps)
            if (!_catalogScanned)
            {
                ScanAccessoryFolder();
                foreach (var kvp in _catalog)
                    _cachedCatalog[kvp.Key] = kvp.Value;
                _catalogScanned = true;
            }
            else
            {
                _catalog.Clear();
                foreach (var kvp in _cachedCatalog)
                    _catalog[kvp.Key] = kvp.Value;
            }

            // Re-equip saved state for this model
            var state = LoadState();
            if (state != null && state.perModel.ContainsKey(_currentModelId))
            {
                foreach (var kvp in state.perModel[_currentModelId])
                {
                    if (kvp.Value)
                        Equip(kvp.Key);
                }
            }

            Debug.Log($"[AccessoryManager] Initialized for {_currentModelId}, catalog={_catalog.Count}");
        }

        /// <summary>Equip an accessory by its catalog ID.</summary>
        public void Equip(string accessoryId)
        {
            if (_animator == null) return;
            if (!_catalog.ContainsKey(accessoryId)) return;
            // Already equipped?
            if (_active.Exists(a => a.id == accessoryId)) return;

            var def = _catalog[accessoryId];
            var bone = _animator.GetBoneTransform(def.bone);
            if (bone == null)
            {
                Debug.LogWarning($"[AccessoryManager] Bone {def.bone} not found for {accessoryId}");
                return;
            }

            // Load the prefab/model
            GameObject obj = LoadAccessoryObject(def);
            if (obj == null) return;

            obj.name = $"Accessory_{accessoryId}";

            var instance = new AccessoryInstance
            {
                id = accessoryId,
                obj = obj,
                targetBone = bone,
                positionOffset = def.positionOffset,
                rotationOffset = Quaternion.Euler(def.rotationOffset),
                scaleOverride = def.scale
            };

            obj.transform.localScale = Vector3.one * def.scale;
            _active.Add(instance);

            SaveEquipState(accessoryId, true);
            OnAccessoryToggled?.Invoke(accessoryId, true);

            Debug.Log($"[AccessoryManager] Equipped: {accessoryId} → {def.bone}");
        }

        /// <summary>Unequip an accessory by its catalog ID.</summary>
        public void Unequip(string accessoryId)
        {
            var idx = _active.FindIndex(a => a.id == accessoryId);
            if (idx < 0) return;

            var instance = _active[idx];
            if (instance.obj != null)
                Destroy(instance.obj);
            _active.RemoveAt(idx);

            SaveEquipState(accessoryId, false);
            OnAccessoryToggled?.Invoke(accessoryId, false);

            Debug.Log($"[AccessoryManager] Unequipped: {accessoryId}");
        }

        /// <summary>Toggle an accessory on/off.</summary>
        public void Toggle(string accessoryId)
        {
            if (_active.Exists(a => a.id == accessoryId))
                Unequip(accessoryId);
            else
                Equip(accessoryId);
        }

        /// <summary>Check if an accessory is currently equipped.</summary>
        public bool IsEquipped(string accessoryId)
        {
            return _active.Exists(a => a.id == accessoryId);
        }

        /// <summary>Get all catalog entry IDs.</summary>
        public IEnumerable<string> GetCatalogIds() => _catalog.Keys;

        /// <summary>Get a catalog definition by ID.</summary>
        public AccessoryDefinition GetDefinition(string id)
        {
            return _catalog.ContainsKey(id) ? _catalog[id] : null;
        }

        public void RemoveAll()
        {
            for (int i = _active.Count - 1; i >= 0; i--)
            {
                if (_active[i].obj != null)
                    Destroy(_active[i].obj);
            }
            _active.Clear();
        }

        // ── Per-frame bone tracking (Mate Engine's smoothness lerp) ──

        private void LateUpdate()
        {
            float dt = Time.deltaTime;
            for (int i = _active.Count - 1; i >= 0; i--)
            {
                var inst = _active[i];
                if (inst.obj == null || inst.targetBone == null)
                {
                    _active.RemoveAt(i);
                    continue;
                }

                // Target position = bone + offset in bone-local space
                Vector3 targetPos = inst.targetBone.TransformPoint(inst.positionOffset);
                Quaternion targetRot = inst.targetBone.rotation * inst.rotationOffset;

                // Smooth lerp toward bone each frame (like Mate Engine's smoothness parameter)
                inst.obj.transform.position = Vector3.Lerp(
                    inst.obj.transform.position, targetPos, positionSmooth * dt);
                inst.obj.transform.rotation = Quaternion.Slerp(
                    inst.obj.transform.rotation, targetRot, rotationSmooth * dt);
            }
        }

        // ── Catalog scanning ────────────────────────────────────

        private void ScanAccessoryFolder()
        {
            _catalog.Clear();

            // Look in StreamingAssets/Accessories/ for JSON definition files
            string folder = Path.Combine(Application.streamingAssetsPath, "Accessories");
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
                Debug.Log($"[AccessoryManager] Created accessories folder: {folder}");
                return;
            }

            var jsonFiles = Directory.GetFiles(folder, "*.json", SearchOption.AllDirectories);
            foreach (var jsonPath in jsonFiles)
            {
                try
                {
                    string json = File.ReadAllText(jsonPath);
                    var def = JsonUtility.FromJson<AccessoryDefinition>(json);
                    if (!string.IsNullOrEmpty(def.id))
                    {
                        def.folderPath = Path.GetDirectoryName(jsonPath);
                        _catalog[def.id] = def;
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[AccessoryManager] Failed to load {jsonPath}: {e.Message}");
                }
            }

            Debug.Log($"[AccessoryManager] Scanned {_catalog.Count} accessories from {folder}");
        }

        private GameObject LoadAccessoryObject(AccessoryDefinition def)
        {
            // Try loading from Resources first (for bundled accessories)
            string resourcePath = $"Accessories/{def.id}";
            var prefab = Resources.Load<GameObject>(resourcePath);
            if (prefab != null)
                return Instantiate(prefab);

            // Fallback: create a lightweight placeholder cube (no collider allocation)
            Debug.Log($"[AccessoryManager] No prefab found for {def.id}, using placeholder cube");
            var go = new GameObject("PlaceholderCube");
            var mf = go.AddComponent<MeshFilter>();
            mf.sharedMesh = Resources.GetBuiltinResource<Mesh>("Cube.fbx");
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            go.transform.localScale = Vector3.one * 0.05f;

            return go;
        }

        // ── Persistence ─────────────────────────────────────────

        private void SaveEquipState(string accessoryId, bool equipped)
        {
            var state = LoadState() ?? new AccessoryState();

            if (!state.perModel.ContainsKey(_currentModelId))
                state.perModel[_currentModelId] = new Dictionary<string, bool>();

            state.perModel[_currentModelId][accessoryId] = equipped;
            _pendingState = state;
            _stateDirty = true;
        }

        private AccessoryState _pendingState;

        private void FlushState()
        {
            if (_pendingState == null) return;
            try
            {
                string json = JsonUtility.ToJson(new AccessoryStateSerializable(_pendingState), true);
                File.WriteAllText(StatePath, json);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[AccessoryManager] Save failed: {e.Message}");
            }
            _stateDirty = false;
        }

        private void OnApplicationQuit()
        {
            if (_stateDirty) FlushState();
        }

        private AccessoryState LoadState()
        {
            if (!File.Exists(StatePath)) return null;

            try
            {
                string json = File.ReadAllText(StatePath);
                var s = JsonUtility.FromJson<AccessoryStateSerializable>(json);
                return s?.ToState();
            }
            catch
            {
                return null;
            }
        }

        // ── Inner types ─────────────────────────────────────────

        private class AccessoryInstance
        {
            public string id;
            public GameObject obj;
            public Transform targetBone;
            public Vector3 positionOffset;
            public Quaternion rotationOffset;
            public float scaleOverride;
        }
    }

    /// <summary>
    /// JSON definition for an accessory (placed in StreamingAssets/Accessories/).
    /// </summary>
    [Serializable]
    public class AccessoryDefinition
    {
        public string id;
        public string displayName;
        public HumanBodyBones bone = HumanBodyBones.Head;
        public Vector3 positionOffset;
        public Vector3 rotationOffset;
        public float scale = 1f;

        [NonSerialized] public string folderPath;
    }

    /// <summary>Per-model accessory equip state (runtime).</summary>
    public class AccessoryState
    {
        public Dictionary<string, Dictionary<string, bool>> perModel
            = new Dictionary<string, Dictionary<string, bool>>();
    }

    /// <summary>Serializable wrapper for AccessoryState (JsonUtility-compatible).</summary>
    [Serializable]
    public class AccessoryStateSerializable
    {
        public List<ModelEntry> models = new List<ModelEntry>();

        public AccessoryStateSerializable() { }

        public AccessoryStateSerializable(AccessoryState state)
        {
            foreach (var kvp in state.perModel)
            {
                var entry = new ModelEntry { modelId = kvp.Key };
                foreach (var acc in kvp.Value)
                    entry.accessories.Add(new AccEntry { id = acc.Key, equipped = acc.Value });
                models.Add(entry);
            }
        }

        public AccessoryState ToState()
        {
            var state = new AccessoryState();
            foreach (var m in models)
            {
                var dict = new Dictionary<string, bool>();
                foreach (var a in m.accessories)
                    dict[a.id] = a.equipped;
                state.perModel[m.modelId] = dict;
            }
            return state;
        }

        [Serializable]
        public class ModelEntry
        {
            public string modelId;
            public List<AccEntry> accessories = new List<AccEntry>();
        }

        [Serializable]
        public class AccEntry
        {
            public string id;
            public bool equipped;
        }
    }
}
