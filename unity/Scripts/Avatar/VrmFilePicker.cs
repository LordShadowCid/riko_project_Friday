using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Opens a native Windows file dialog to select a .vrm file.
    /// Uses Win32 GetOpenFileName (comdlg32.dll) — no external packages needed.
    /// Based on Mate-Engine VRMLoader.cs file dialog pattern, but uses P/Invoke
    /// instead of StandaloneFileBrowser to avoid the external dependency.
    /// </summary>
    public static class VrmFilePicker
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        [DllImport("comdlg32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern bool GetOpenFileName(ref OpenFileName ofn);

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct OpenFileName
        {
            public int       lStructSize;
            public IntPtr    hwndOwner;
            public IntPtr    hInstance;
            public string    lpstrFilter;
            public string    lpstrCustomFilter;
            public int       nMaxCustFilter;
            public int       nFilterIndex;
            public IntPtr    lpstrFile;
            public int       nMaxFile;
            public string    lpstrFileTitle;
            public int       nMaxFileTitle;
            public string    lpstrInitialDir;
            public string    lpstrTitle;
            public int       Flags;
            public short     nFileOffset;
            public short     nFileExtension;
            public string    lpstrDefExt;
            public IntPtr    lCustData;
            public IntPtr    lpfnHook;
            public string    lpTemplateName;
            public IntPtr    pvReserved;
            public int       dwReserved;
            public int       FlagsEx;
        }

        private const int OFN_PATHMUSTEXIST  = 0x00000800;
        private const int OFN_FILEMUSTEXIST  = 0x00001000;
        private const int OFN_NOCHANGEDIR    = 0x00000008;
        private const int MAX_PATH           = 4096;
#endif

        /// <summary>
        /// Opens a native Windows file dialog filtered to .vrm files.
        /// Returns the selected absolute path, or null if cancelled.
        /// In the Editor, falls back to EditorUtility.OpenFilePanel.
        /// </summary>
        public static string OpenVrmFileDialog()
        {
#if UNITY_EDITOR
            string path = UnityEditor.EditorUtility.OpenFilePanel(
                "Select VRM Model", "", "vrm");
            return string.IsNullOrEmpty(path) ? null : path;
#elif UNITY_STANDALONE_WIN
            IntPtr buffer = Marshal.AllocHGlobal(MAX_PATH * 2);
            try
            {
                Marshal.Copy(new byte[MAX_PATH * 2], 0, buffer, MAX_PATH * 2);

                var ofn = new OpenFileName();
                ofn.lStructSize = Marshal.SizeOf(ofn);
                ofn.hwndOwner = IntPtr.Zero;
                ofn.lpstrFilter = "VRM Models (*.vrm)\0*.vrm\0All Files (*.*)\0*.*\0\0";
                ofn.lpstrFile = buffer;
                ofn.nMaxFile = MAX_PATH;
                ofn.lpstrTitle = "Select VRM Model";
                ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
                ofn.lpstrDefExt = "vrm";

                string defaultDir = System.IO.Path.Combine(
                    Application.streamingAssetsPath, "Models");
                if (System.IO.Directory.Exists(defaultDir))
                    ofn.lpstrInitialDir = defaultDir;

                if (GetOpenFileName(ref ofn))
                {
                    string selected = Marshal.PtrToStringUni(buffer);
                    Debug.Log($"[VrmFilePicker] Selected: {selected}");
                    return selected;
                }

                return null;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
#else
            Debug.LogWarning("[VrmFilePicker] File picker not supported on this platform.");
            return null;
#endif
        }
    }
}
