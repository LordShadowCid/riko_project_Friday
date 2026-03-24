using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;

namespace Annabeth.Core
{
    /// <summary>
    /// System tray icon with right-click context menu (Show/Hide, Settings, Sleep, Quit).
    /// Uses Win32 Shell_NotifyIcon — Windows standalone only.
    /// </summary>
    public class SystemTrayController : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        // ── P/Invoke ────────────────────────────────────────────
        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern bool Shell_NotifyIcon(uint msg, ref NOTIFYICONDATA data);
        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr ExtractIcon(IntPtr hInst, string file, int index);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        static extern ushort RegisterClassW(ref WNDCLASSW wc);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr CreateWindowExW(uint exStyle, string cls, string name, uint style,
            int x, int y, int w, int h, IntPtr parent, IntPtr menu, IntPtr inst, IntPtr param);
        [DllImport("user32.dll")] static extern bool DestroyWindow(IntPtr hWnd);
        [DllImport("user32.dll")] static extern IntPtr DefWindowProcW(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
        [DllImport("user32.dll")] static extern int GetMessageW(out MSG msg, IntPtr hWnd, uint min, uint max);
        [DllImport("user32.dll")] static extern bool TranslateMessage(ref MSG msg);
        [DllImport("user32.dll")] static extern IntPtr DispatchMessageW(ref MSG msg);
        [DllImport("user32.dll")] static extern bool PostMessageW(IntPtr hWnd, uint msg, IntPtr wp, IntPtr lp);
        [DllImport("user32.dll")] static extern void PostQuitMessage(int code);
        [DllImport("user32.dll")] static extern IntPtr CreatePopupMenu();
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        static extern bool AppendMenuW(IntPtr hMenu, uint flags, uint id, string text);
        [DllImport("user32.dll")]
        static extern int TrackPopupMenuEx(IntPtr hMenu, uint flags, int x, int y, IntPtr hWnd, IntPtr tpm);
        [DllImport("user32.dll")] static extern bool DestroyMenu(IntPtr hMenu);
        [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr hWnd);
        [DllImport("user32.dll")] static extern bool GetCursorPos(out POINT pt);
        [DllImport("user32.dll")] static extern IntPtr LoadIcon(IntPtr hInst, IntPtr name);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr GetModuleHandle(string name);

        // ── Structs ─────────────────────────────────────────────
        delegate IntPtr WndProcDelegate(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct WNDCLASSW
        {
            public uint style; public WndProcDelegate lpfnWndProc;
            public int cbClsExtra, cbWndExtra;
            public IntPtr hInstance, hIcon, hCursor, hbrBackground;
            public string lpszMenuName, lpszClassName;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct NOTIFYICONDATA
        {
            public int cbSize; public IntPtr hWnd; public uint uID; public uint uFlags;
            public uint uCallbackMessage; public IntPtr hIcon;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string szTip;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct MSG { public IntPtr hwnd; public uint message; public IntPtr wParam, lParam; public uint time; public POINT pt; }

        [StructLayout(LayoutKind.Sequential)]
        struct POINT { public int X, Y; }

        // ── Constants ───────────────────────────────────────────
        const uint NIM_ADD = 0, NIM_DELETE = 2;
        const uint NIF_MESSAGE = 1, NIF_ICON = 2, NIF_TIP = 4;
        const uint WM_USER = 0x0400, WM_TRAYICON = WM_USER + 1;
        const uint WM_LBUTTONDBLCLK = 0x0203, WM_RBUTTONUP = 0x0205;
        const uint WM_COMMAND = 0x0111, WM_CLOSE = 0x0010;
        const uint MF_STRING = 0, MF_SEPARATOR = 0x0800;
        const uint TPM_RETURNCMD = 0x0100;
        static readonly IntPtr HWND_MESSAGE = new IntPtr(-3);
        static readonly IntPtr IDI_APPLICATION = new IntPtr(32512);

        const uint CMD_SHOW_HIDE = 1, CMD_SETTINGS = 2, CMD_SLEEP = 3, CMD_QUIT = 4;

        // ── State ───────────────────────────────────────────────
        static readonly ConcurrentQueue<uint> _commands = new();
        static WndProcDelegate _wndProcRef;
        static IntPtr _msgWnd;
        Thread _thread;
        bool _hidden;

        void Awake()
        {
            Application.wantsToQuit += OnWantsToQuit;
        }

        void Start()
        {
            _wndProcRef = TrayWndProc;
            _thread = new Thread(RunTrayThread) { IsBackground = true, Name = "AnnabethTray" };
            _thread.Start();
        }

        void Update()
        {
            while (_commands.TryDequeue(out uint cmd))
                HandleCommand(cmd);
        }

        void OnApplicationQuit()
        {
            Application.wantsToQuit -= OnWantsToQuit;
            if (_msgWnd != IntPtr.Zero)
                PostMessageW(_msgWnd, WM_CLOSE, IntPtr.Zero, IntPtr.Zero);
        }

        bool OnWantsToQuit()
        {
            if (SettingsManager.Instance != null && SettingsManager.Instance.data.minimizeToTray)
            {
                ToggleVisibility();
                return false;
            }
            return true;
        }

        // ── Tray Thread ─────────────────────────────────────────
        void RunTrayThread()
        {
            var hInst = GetModuleHandle(null);
            var wc = new WNDCLASSW
            {
                lpfnWndProc = _wndProcRef,
                hInstance = hInst,
                lpszClassName = "AnnabethTray"
            };
            RegisterClassW(ref wc);

            _msgWnd = CreateWindowExW(0, "AnnabethTray", "", 0, 0, 0, 0, 0,
                HWND_MESSAGE, IntPtr.Zero, hInst, IntPtr.Zero);

            IntPtr icon = IntPtr.Zero;
            try
            {
                string exe = System.Diagnostics.Process.GetCurrentProcess().MainModule?.FileName;
                if (!string.IsNullOrEmpty(exe))
                    icon = ExtractIcon(hInst, exe, 0);
            }
            catch { /* fallback below */ }
            if (icon == IntPtr.Zero)
                icon = LoadIcon(IntPtr.Zero, IDI_APPLICATION);

            var nid = new NOTIFYICONDATA
            {
                cbSize = Marshal.SizeOf<NOTIFYICONDATA>(),
                hWnd = _msgWnd, uID = 1,
                uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP,
                uCallbackMessage = WM_TRAYICON,
                hIcon = icon, szTip = "Annabeth"
            };
            Shell_NotifyIcon(NIM_ADD, ref nid);

            while (GetMessageW(out MSG msg, IntPtr.Zero, 0, 0) > 0)
            {
                TranslateMessage(ref msg);
                DispatchMessageW(ref msg);
            }

            Shell_NotifyIcon(NIM_DELETE, ref nid);
            DestroyWindow(_msgWnd);
            _msgWnd = IntPtr.Zero;
        }

        static IntPtr TrayWndProc(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam)
        {
            if (msg == WM_TRAYICON)
            {
                uint mouse = (uint)(lParam.ToInt64() & 0xFFFF);
                if (mouse == WM_RBUTTONUP) ShowContextMenu(hWnd);
                else if (mouse == WM_LBUTTONDBLCLK) _commands.Enqueue(CMD_SHOW_HIDE);
                return IntPtr.Zero;
            }
            if (msg == WM_COMMAND)
            {
                _commands.Enqueue((uint)(wParam.ToInt64() & 0xFFFF));
                return IntPtr.Zero;
            }
            if (msg == WM_CLOSE)
            {
                PostQuitMessage(0);
                return IntPtr.Zero;
            }
            return DefWindowProcW(hWnd, msg, wParam, lParam);
        }

        static void ShowContextMenu(IntPtr hWnd)
        {
            IntPtr menu = CreatePopupMenu();
            AppendMenuW(menu, MF_STRING, CMD_SHOW_HIDE, "Show / Hide");
            AppendMenuW(menu, MF_STRING, CMD_SETTINGS, "Settings");
            AppendMenuW(menu, MF_STRING, CMD_SLEEP, "Sleep / Wake");
            AppendMenuW(menu, MF_SEPARATOR, 0, null);
            AppendMenuW(menu, MF_STRING, CMD_QUIT, "Quit");

            GetCursorPos(out POINT pt);
            SetForegroundWindow(hWnd);
            int cmd = TrackPopupMenuEx(menu, TPM_RETURNCMD, pt.X, pt.Y, hWnd, IntPtr.Zero);
            DestroyMenu(menu);

            if (cmd > 0) _commands.Enqueue((uint)cmd);
        }

        // ── Command Handlers (Main Thread) ──────────────────────
        void HandleCommand(uint cmd)
        {
            switch (cmd)
            {
                case CMD_SHOW_HIDE:
                    ToggleVisibility();
                    break;
                case CMD_SETTINGS:
                    FindFirstObjectByType<UI.RadialMenu>()?.OpenSettings();
                    break;
                case CMD_SLEEP:
                    FindFirstObjectByType<SleepController>()?.ToggleSleep();
                    break;
                case CMD_QUIT:
                    Application.wantsToQuit -= OnWantsToQuit;
                    Application.Quit();
                    break;
            }
        }

        void ToggleVisibility()
        {
            _hidden = !_hidden;
            var vrm = FindFirstObjectByType<UniVRM10.Vrm10Instance>();
            if (vrm != null) vrm.gameObject.SetActive(!_hidden);
            Debug.Log($"[SystemTrayController] Avatar {(_hidden ? "hidden" : "shown")}");
        }
#else
        void Start()
        {
            Debug.Log("[SystemTrayController] System tray available only in Windows standalone builds.");
        }
#endif
    }
}
